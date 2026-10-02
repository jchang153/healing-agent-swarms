"""Offline tests for shared-board access and serialized solver turns."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
from inspect_ai.tool import tool

from experiments.swarm_tools import FairScheduler, SharedBoard, swarm_adapter


class SharedBoardTests(unittest.TestCase):
    def test_posts_are_bounded_copied_and_reads_are_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "board.jsonl"
            board = SharedBoard("group-a", True, log, max_messages=2,
                                max_text_length=12)
            first = board.post("agent-1", "hello")
            board.post("agent-2", "world")
            first["text"] = "tampered"
            self.assertEqual([m["text"] for m in board.read("agent-3")],
                             ["hello", "world"])
            with self.assertRaises(ValueError):
                board.post("agent-3", "x")
            with self.assertRaises(ValueError):
                board.post("agent-3", "x" * 13)

            events = [json.loads(line) for line in log.read_text().splitlines()]
            read_event = next(event for event in events if event["event"] == "board_read")
            self.assertEqual(read_event["exposed_message_ids"], [0, 1])
            self.assertNotIn("hello", json.dumps(read_event))

    def test_disabled_board_rejects_reads_and_posts_without_creating_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "unused.jsonl"
            board = SharedBoard("group-a", False, log)
            with self.assertRaises(RuntimeError):
                board.post("agent-1", "secret")
            with self.assertRaises(RuntimeError):
                board.read("agent-1")
            self.assertFalse(log.exists())

    def test_separate_groups_have_isolated_histories_and_audit_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = SharedBoard("left", True, root / "left.jsonl")
            right = SharedBoard("right", True, root / "right.jsonl")
            left.post("agent-1", "left-only")
            right.post("agent-1", "right-only")
            self.assertEqual([m["text"] for m in left.read("agent-2")], ["left-only"])
            self.assertEqual([m["text"] for m in right.read("agent-2")], ["right-only"])
            left_rows = [json.loads(x) for x in (root / "left.jsonl").read_text().splitlines()]
            right_rows = [json.loads(x) for x in (root / "right.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["group_id"] == "left" for row in left_rows))
            self.assertTrue(all(row["group_id"] == "right" for row in right_rows))

    def test_invalid_author_and_cursor_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            with self.assertRaises(ValueError):
                board.post("", "text")
            for cursor in (-1, True, "0"):
                with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                    board.read("reader", cursor)


class FairSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_turns_are_exclusive_and_turn_cap_is_per_agent(self):
        scheduler = FairScheduler(["a", "b", "c"], max_turns=1, seconds=2)
        self.assertTrue(await scheduler.acquire("a"))
        order = []

        async def contender(agent):
            acquired = await scheduler.acquire(agent)
            if acquired:
                order.append(agent)
                await scheduler.release(agent)

        second = asyncio.create_task(contender("b"))
        await asyncio.sleep(0)
        third = asyncio.create_task(contender("c"))
        await asyncio.sleep(0)
        await scheduler.release("a")
        await asyncio.gather(second, third)
        self.assertEqual(order, ["b", "c"])
        self.assertFalse(await scheduler.acquire("a"))
        self.assertEqual(scheduler.snapshot()["a"]["turns"], 1)

    async def test_finishing_owner_releases_turn_for_next_agent(self):
        scheduler = FairScheduler(["owner", "next"], 2, 2)
        self.assertTrue(await scheduler.acquire("owner"))
        next_agent = asyncio.create_task(scheduler.acquire("next"))
        await asyncio.sleep(0)
        await scheduler.finish("owner", "finished")
        self.assertTrue(await asyncio.wait_for(next_agent, timeout=0.5))
        self.assertEqual(scheduler.snapshot()["owner"]["status"], "finished")
        await scheduler.release("next")

    async def test_finishing_queued_agent_terminates_its_pending_acquire(self):
        scheduler = FairScheduler(["owner", "departed", "next"], 2, 2)
        self.assertTrue(await scheduler.acquire("owner"))
        departed = asyncio.create_task(scheduler.acquire("departed"))
        for _ in range(20):
            if "departed" in scheduler._queue:
                break
            await asyncio.sleep(0)
        self.assertIn("departed", scheduler._queue)
        next_agent = asyncio.create_task(scheduler.acquire("next"))
        for _ in range(20):
            if "next" in scheduler._queue:
                break
            await asyncio.sleep(0)
        await scheduler.finish("departed", "cancelled")
        self.assertFalse(await asyncio.wait_for(departed, timeout=0.5))
        self.assertEqual(scheduler.snapshot()["departed"]["status"], "cancelled")
        await scheduler.release("owner")
        self.assertTrue(await asyncio.wait_for(next_agent, timeout=0.5))
        await scheduler.release("next")

    async def test_deadline_returns_false_and_releases_waiter_without_deadlock(self):
        scheduler = FairScheduler(["owner", "waiter"], 3, 0.03)
        self.assertTrue(await scheduler.acquire("owner"))
        self.assertFalse(await asyncio.wait_for(scheduler.acquire("waiter"), timeout=0.5))
        self.assertEqual(scheduler.snapshot()["waiter"]["status"], "time_limit")
        await scheduler.release("owner")

    async def test_unknown_agent_is_rejected(self):
        scheduler = FairScheduler(["a"], 1, 1)
        with self.assertRaises(KeyError):
            await scheduler.acquire("stranger")


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def model_output(content="mock response", stop_reason="stop"):
        return ModelOutput(
            model="mockllm/offline",
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=content),
                stop_reason=stop_reason,
            )],
        )

    async def test_direct_model_interception_adds_board_after_upstream_tool_setup_and_stops_at_quota(self):
        observed_tools = []
        calls = []

        async def original_generate(*, input, tools, **kwargs):
            observed_tools.append({item.__registry_info__.name for item in tools})
            calls.append(input)
            return self.model_output()

        model = SimpleNamespace(generate=original_generate)

        async def inner(state, _generate):
            # basic_agent sets its tools then calls model.generate directly.
            state.tools = [upstream_tool]
            first = await model.generate(input=["first"], tools=state.tools)
            self.assertEqual(first.choices[0].message.content, "mock response")
            second = await model.generate(input=["second"], tools=state.tools)
            self.assertEqual(second.choices[0].stop_reason, "stop")
            return state

        @tool(name="upstream_tool")
        async def upstream_tool():
            """Tool installed by the upstream agent."""
            return "ok"

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["agent"], max_turns=1, seconds=1)
            adapted = swarm_adapter(inner, "agent", board, scheduler,
                                    model_instance=model)
            state = SimpleNamespace(tools=[], completed=False, metadata={})

            result = await adapted(state, lambda current: asyncio.sleep(0, result=current))

            self.assertIs(result, state)
            self.assertEqual(calls, [["first"]])
            self.assertEqual(len(observed_tools), 1)
            self.assertTrue({"upstream_tool", "board_read", "board_post"} <= observed_tools[0])
            self.assertIs(model.generate, original_generate)
            self.assertEqual(result.metadata["swarm_stop_reason"], "limit")
            self.assertEqual(result.metadata["swarm"]["turns"], 1)

    async def test_direct_model_budget_402_marks_state_complete_and_restores_generate(self):
        stop_checks = 0
        calls = []

        async def original_generate(*, input, tools, **kwargs):
            calls.append(input)
            error = RuntimeError("pre-dispatch budget rejection")
            error.status_code = 402
            raise error

        model = SimpleNamespace(generate=original_generate)

        def stop_check(_agent):
            nonlocal stop_checks
            stop_checks += 1
            # Allow preflight and acquisition checks, then expose the gateway's
            # concrete per-agent dollar stop after the model rejects dispatch.
            return "dollars" if stop_checks >= 3 else None

        async def inner(state, _generate):
            state.tools = [upstream_tool]
            output = await model.generate(input=["budget-limited"], tools=state.tools)
            self.assertEqual(output.choices[0].stop_reason, "stop")
            return state

        @tool(name="upstream_tool")
        async def upstream_tool():
            """Tool installed by the upstream agent."""
            return "ok"

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["agent"], max_turns=2, seconds=1)
            adapted = swarm_adapter(inner, "agent", board, scheduler,
                                    stop_check=stop_check, model_instance=model)
            state = SimpleNamespace(tools=[], completed=False, metadata={})

            result = await adapted(state, lambda current: asyncio.sleep(0, result=current))

            self.assertIs(result, state)
            self.assertTrue(result.completed)
            self.assertEqual(calls, [["budget-limited"]])
            self.assertEqual(result.metadata["swarm_stop_reason"], "dollars")
            self.assertEqual(result.metadata["swarm"]["status"], "dollars")
            self.assertEqual(result.metadata["swarm"]["turns"], 1)
            self.assertIs(model.generate, original_generate)

    async def test_adapter_preserves_tools_scopes_board_author_and_releases_on_cancel(self):
        @tool(name="upstream_tool")
        async def upstream_tool():
            """Existing solver tool."""
            return "ok"

        async def inner(state, generate):
            self.assertIn(upstream_tool, state.tools)
            # The real upstream agent installs its tools immediately before
            # generation; the adapter must retain them and add board tools then.
            state.tools = [upstream_tool]

            async def generation_with_checks(current):
                names = {item.__registry_info__.name for item in current.tools}
                self.assertTrue({"upstream_tool", "board_read", "board_post"} <= names)
                raise asyncio.CancelledError()

            await generate(state, generation_with_checks)
            return state

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["trusted-agent"], 3, 1)
            adapted = swarm_adapter(inner, "trusted-agent", board, scheduler)
            state = SimpleNamespace(tools=[upstream_tool], completed=False)
            async def generate(_state, callback):
                return await callback(_state)

            with self.assertRaises(asyncio.CancelledError):
                await adapted(state, generate)
            snapshot = scheduler.snapshot()["trusted-agent"]
            self.assertEqual(snapshot["status"], "cancelled")
            self.assertEqual(snapshot["turns"], 1)
            # A second scheduler client is not needed: successful post tooling
            # uses the trusted closure identity supplied by the adapter.
            tools = {item.__registry_info__.name: item for item in state.tools}
            await tools["board_post"]("posted by closure")
            self.assertEqual(board.read("reviewer")[0]["author"], "trusted-agent")

    async def test_adapter_marks_turn_limit_on_returned_state(self):
        async def inner(state, generate):
            await generate(state)
            return await generate(state)

        async def generate(state):
            return state

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["a"], 1, 1)
            adapted = swarm_adapter(inner, "a", board, scheduler)
            state = SimpleNamespace(tools=[], completed=False, metadata={})
            result = await adapted(state, generate)
            self.assertTrue(result.completed)
            self.assertEqual(result.metadata["swarm_stop_reason"], "limit")
            self.assertEqual(result.metadata["swarm"]["status"], "limit")

    async def test_stop_check_prevents_generation_and_records_reason(self):
        async def inner(state, generate):
            return await generate(state)

        async def generate(state):
            self.fail("generation must not run when the stop check fires")

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", True, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["a"], 1, 1)
            adapted = swarm_adapter(inner, "a", board, scheduler,
                                    stop_check=lambda _agent: "budget")
            state = SimpleNamespace(tools=[], completed=False, metadata={})
            result = await adapted(state, generate)
            self.assertTrue(result.completed)
            self.assertEqual(result.metadata["swarm_stop_reason"], "budget")
            self.assertEqual(result.metadata["swarm"]["status"], "budget")

    async def test_unexpected_generation_error_propagates_and_marks_error(self):
        async def inner(state, generate):
            return await generate(state)

        async def generate(_state):
            raise RuntimeError("unexpected failure")

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", False, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["a"], 1, 1)
            adapted = swarm_adapter(inner, "a", board, scheduler)
            state = SimpleNamespace(tools=[], completed=False, metadata={})
            with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
                await adapted(state, generate)
            self.assertEqual(scheduler.snapshot()["a"]["status"], "error")

    async def test_adapter_with_disabled_board_keeps_upstream_toolset_unchanged(self):
        async def inner(state, generate):
            return state

        with tempfile.TemporaryDirectory() as directory:
            board = SharedBoard("g", False, Path(directory) / "board.jsonl")
            scheduler = FairScheduler(["a"], 1, 1)
            adapted = swarm_adapter(inner, "a", board, scheduler)
            original = object()
            state = SimpleNamespace(tools=[original], completed=False)
            result = await adapted(state, lambda current: asyncio.sleep(0, result=current))
            self.assertIs(result, state)
            self.assertEqual(state.tools, [original])


if __name__ == "__main__":
    unittest.main()
