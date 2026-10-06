"""Inspect adapter for a shared board and fair, serialized generation turns.

The adapter wraps an already initialized Inspect solver. It does not change task
construction, scoring, or upstream solver configuration.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Awaitable, Callable, Literal

from inspect_ai.solver import TaskState, Solver, solver
from inspect_ai.tool import tool


@dataclass(frozen=True)
class _Message:
    id: int
    author: str
    text: str
    timestamp: str
    parent_id: int | None = None
    thread_id: int | None = None
    relation: str = "comment"


class SharedBoard:
    """A bounded board whose authors are supplied by the trusted caller."""

    def __init__(self, group_id: str, enabled: bool, log_path: str | Path,
                 *, max_messages: int = 1000, max_text_length: int = 2000,
                 structure: str = "flat") -> None:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be a positive integer")
        if isinstance(max_text_length, bool) or not isinstance(max_text_length, int) or max_text_length < 1:
            raise ValueError("max_text_length must be a positive integer")
        self.group_id = group_id
        if structure not in ("flat", "threaded"):
            raise ValueError("structure must be flat or threaded")
        self.structure = structure
        self.enabled = enabled
        self.log_path = Path(log_path)
        self.max_messages = max_messages
        self.max_text_length = max_text_length
        self._messages: list[_Message] = []
        self._delivered: dict[str, set[int]] = {}
        self._lock = RLock()

    def _log(self, event: str, **fields: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), group_id=self.group_id,
                   event=event, **fields)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _payload(self, item: _Message) -> dict[str, Any]:
        result = asdict(item)
        if self.structure == "flat":
            return {k: result[k] for k in ("id", "author", "text", "timestamp")}
        return result

    def post(self, agent_id: str, text: str, *, parent_id: int | None = None,
             relation: str = "comment") -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(text, str) or len(text) > self.max_text_length:
            raise ValueError(f"text must be a string up to {self.max_text_length} characters")
        with self._lock:
            if relation not in ("comment", "agree", "disagree", "question"):
                raise ValueError("Unknown reply relation")
            if self.structure == "flat" and (parent_id is not None or relation != "comment"):
                raise ValueError("Replies require threaded structure")
            if parent_id is None and relation != "comment":
                raise ValueError("A relation requires a reply parent")
            if parent_id is not None:
                if isinstance(parent_id, bool) or not isinstance(parent_id, int) or not 0 <= parent_id < len(self._messages):
                    raise ValueError("parent_id must identify an existing message")
                parent = self._messages[parent_id]
                if parent.author != agent_id and parent_id not in self._delivered.get(agent_id, set()):
                    raise ValueError("Read the parent message before replying")
            if len(self._messages) >= self.max_messages:
                raise ValueError("Board capacity reached")
            item = _Message(id=len(self._messages), author=agent_id, text=text,
                            timestamp=datetime.now(timezone.utc).isoformat(), parent_id=parent_id,
                            thread_id=(self._messages[parent_id].thread_id if parent_id is not None else len(self._messages)),
                            relation=relation)
            self._messages.append(item)
            result = self._payload(item)
            if self.structure == "threaded":
                self._log("board_post", message=result,
                          read_before_post_ids=sorted(i for i in self._delivered.get(agent_id, set())
                                                      if self._messages[i].author != agent_id))
            else:
                self._log("board_post", message=result)
            return result

    def read(self, agent_id: str, since: int = 0) -> list[dict[str, Any]]:
        if self.structure == "threaded":
            raise ValueError("Use read_threads for a threaded board")
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if isinstance(since, bool) or not isinstance(since, int) or since < 0:
            raise ValueError("since must be a non-negative integer message id")
        with self._lock:
            result = [self._payload(item) for item in self._messages if item.id >= since]
            self._delivered.setdefault(agent_id, set()).update(item["id"] for item in result)
            self._log("board_read", agent=agent_id, since=since,
                      exposed_message_ids=[item["id"] for item in result])
            return result

    def _thread_access(self, agent_id: str) -> None:
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        if self.structure != "threaded":
            raise ValueError("This operation requires a threaded board")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

    def list_threads(self, agent_id: str, unread_only: bool = False) -> list[dict[str, Any]]:
        """Metadata only: listing does not deliver message bodies or clear unread state."""
        self._thread_access(agent_id)
        if not isinstance(unread_only, bool):
            raise ValueError("unread_only must be a bool")
        with self._lock:
            unread = {m["id"] for m in self.unread(agent_id)}
            result = []
            for root in self._messages:
                if root.parent_id is not None:
                    continue
                members = [m for m in self._messages if m.thread_id == root.id]
                pending = [m.id for m in members if m.id in unread]
                if unread_only and not pending:
                    continue
                result.append(dict(thread_id=root.id, author=root.author, timestamp=root.timestamp,
                                   reply_count=len(members)-1, unread_message_ids=pending))
            self._log("board_list_threads", agent=agent_id, unread_only=unread_only,
                      listed_thread_ids=[r["thread_id"] for r in result], exposed_message_ids=[])
            return result

    def read_threads(self, agent_id: str, thread_id: int | None = None,
                     unread_only: bool = True) -> list[dict[str, Any]]:
        """Return complete trees, including ancestors of unread replies."""
        self._thread_access(agent_id)
        if not isinstance(unread_only, bool):
            raise ValueError("unread_only must be a bool")
        with self._lock:
            if thread_id is not None:
                if isinstance(thread_id, bool) or not isinstance(thread_id, int) or not 0 <= thread_id < len(self._messages) or self._messages[thread_id].parent_id is not None:
                    raise ValueError("thread_id must identify a top-level post")
                roots = {thread_id}
            else:
                roots = {m["thread_id"] for m in self.unread(agent_id)} if unread_only else {m.thread_id for m in self._messages}
            selected = [m for m in self._messages if m.thread_id in roots]
            nodes = {m.id: dict(self._payload(m), replies=[]) for m in selected}
            result = []
            for m in selected:
                if m.parent_id is None:
                    result.append(nodes[m.id])
                else:
                    nodes[m.parent_id]["replies"].append(nodes[m.id])
            ids = [m.id for m in selected]
            self._delivered.setdefault(agent_id, set()).update(ids)
            self._log("board_read", agent=agent_id, thread_id=thread_id, unread_only=unread_only,
                      exposed_message_ids=ids)
            return result

    def unread(self, agent_id: str) -> list[dict[str, Any]]:
        """Messages by other agents that ``agent_id`` has not yet received through ``read``."""
        with self._lock:
            delivered = self._delivered.get(agent_id, set())
            return [self._payload(item) for item in self._messages
                    if item.author != agent_id and item.id not in delivered]

    def deliver_unread(self, agent_id: str) -> list[dict[str, Any]]:
        """Hand ``agent_id`` its unread messages (push delivery) and mark them delivered."""
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        with self._lock:
            pending = self.unread(agent_id)
            if pending:
                self._delivered.setdefault(agent_id, set()).update(m["id"] for m in pending)
                self._log("board_push", agent=agent_id, exposed_message_ids=[m["id"] for m in pending])
            return pending

    def notice(self, agent_id: str) -> str:
        """Per-turn status text for ``agent_id``; logs which unread IDs it reported."""
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        with self._lock:
            pending = self.unread(agent_id)
            self._log("board_notice", agent=agent_id, unread_message_ids=[m["id"] for m in pending])
        if not pending:
            return "[Message board status] No unread messages on the shared board."
        authors = ", ".join(sorted({m["author"] for m in pending}))
        count = len(pending)
        return (f"[Message board status] You have {count} unread message{'s' if count != 1 else ''} "
                f"on the shared board from {authors}. Use board_read to read {'them' if count != 1 else 'it'}.")


class FairScheduler:
    """FIFO scheduler allowing up to ``max_concurrent`` generate calls at a time."""

    def __init__(self, agent_ids: list[str] | tuple[str, ...], max_turns: int,
                 seconds: float, max_concurrent: int = 1) -> None:
        ids = tuple(agent_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("agent_ids must contain unique, non-empty strings")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        self.agent_ids = ids
        self.max_concurrent = max_concurrent
        self.max_turns = max_turns
        self.seconds = float(seconds)
        self._deadline: float | None = None
        self._owners: set[str] = set()
        self._queue: deque[str] = deque()
        self._condition = asyncio.Condition()
        self._turns = {agent: 0 for agent in ids}
        self._status = {agent: "waiting" for agent in ids}
        self._api_times = {agent: [] for agent in ids}
        self._tool_times = {agent: [] for agent in ids}

    def start(self) -> None:
        if self._deadline is None:
            self._deadline = asyncio.get_running_loop().time() + self.seconds

    async def acquire(self, agent_id: str) -> bool:
        if agent_id not in self._turns:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        self.start()
        async with self._condition:
            if self._turns[agent_id] >= self.max_turns:
                self._status[agent_id] = "limit"
                return False
            if agent_id not in self._queue and agent_id not in self._owners:
                self._queue.append(agent_id)
            if self._status[agent_id] in ("waiting", "running"):
                self._status[agent_id] = "waiting"
            while True:
                if self._status[agent_id] not in ("waiting", "running"):
                    self._remove_waiter(agent_id)
                    return False
                remaining = (self._deadline or 0) - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._remove_waiter(agent_id)
                    self._status[agent_id] = "time_limit"
                    self._condition.notify_all()
                    return False
                if len(self._owners) < self.max_concurrent and self._queue and self._queue[0] == agent_id:
                    self._queue.popleft()
                    self._owners.add(agent_id)
                    self._status[agent_id] = "running"
                    self._turns[agent_id] += 1
                    return True
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    self._remove_waiter(agent_id)
                    self._status[agent_id] = "time_limit"
                    self._condition.notify_all()
                    return False

    def _remove_waiter(self, agent_id: str) -> None:
        try:
            self._queue.remove(agent_id)
        except ValueError:
            pass

    async def release(self, agent_id: str) -> None:
        async with self._condition:
            self._owners.discard(agent_id)
            if self._status.get(agent_id) == "running":
                self._status[agent_id] = "waiting"
            self._condition.notify_all()

    async def finish(self, agent_id: str, status: str = "finished") -> None:
        if agent_id not in self._turns:
            return
        async with self._condition:
            self._remove_waiter(agent_id)
            self._owners.discard(agent_id)
            # Keep a quota/deadline result set by acquire() visible after the
            # wrapped solver returns its best available state for scoring.
            if self._status[agent_id] in ("waiting", "running"):
                self._status[agent_id] = status
            self._condition.notify_all()

    def record_api_time(self, agent_id: str, seconds: float) -> None:
        self._api_times[agent_id].append(max(0.0, float(seconds)))

    def record_tool_time(self, agent_id: str, tool_name: str, seconds: float) -> None:
        self._tool_times[agent_id].append({"tool": tool_name, "seconds": max(0.0, float(seconds))})

    def snapshot(self) -> dict[str, Any]:
        return {agent: {"turns": self._turns[agent], "status": self._status[agent],
                        "generation_times_seconds": list(self._api_times[agent]),
                        "tool_times": list(self._tool_times[agent])}
                for agent in self.agent_ids}


def _board_tools(board: SharedBoard, agent_id: str,
                 scheduler: FairScheduler) -> list[Any]:
    if board.structure == "threaded":
        return _threaded_board_tools(board, agent_id, scheduler)
    @tool(name="board_read")
    def board_read() -> Any:
        async def read(since: int = 0) -> str:
            """Read shared messages.

            Args:
                since: Return messages with IDs greater than or equal to this cursor.
            """
            started = time.monotonic()
            try:
                return json.dumps(board.read(agent_id, since), ensure_ascii=False)
            finally:
                scheduler.record_tool_time(agent_id, "board_read", time.monotonic() - started)
        return read

    @tool(name="board_post")
    def board_post() -> Any:
        async def post(text: str) -> str:
            """Post a message to the shared board.

            Args:
                text: Message text, up to 2000 characters.
            """
            started = time.monotonic()
            try:
                return json.dumps(board.post(agent_id, text), ensure_ascii=False)
            finally:
                scheduler.record_tool_time(agent_id, "board_post", time.monotonic() - started)
        return post

    return [board_read(), board_post()]


def _threaded_board_tools(board: SharedBoard, agent_id: str,
                          scheduler: FairScheduler) -> list[Any]:
    def invoke(name, method, *args, **kwargs):
        started = time.monotonic()
        try:
            return json.dumps(method(agent_id, *args, **kwargs), ensure_ascii=False)
        finally:
            scheduler.record_tool_time(agent_id, name, time.monotonic() - started)

    @tool(name="board_list_threads")
    def list_threads():
        async def listing(unread_only: bool = False) -> str:
            """List thread IDs, authors, times, reply counts and unread IDs; no message bodies.

            Args:
                unread_only: Only list threads containing unread peer messages.
            """
            return invoke("board_list_threads", board.list_threads, unread_only)
        return listing

    @tool(name="board_read")
    def read_threads():
        async def read(thread_id: int | None = None, unread_only: bool = True) -> str:
            """Read nested post/reply trees. No cursor is needed; reads clear delivered unread messages.

            Args:
                thread_id: Read this top-level post's complete thread; overrides unread_only.
                unread_only: Without a thread ID, read threads with unread messages; false reads all threads.
            """
            return invoke("board_read", board.read_threads, thread_id, unread_only)
        return read

    @tool(name="board_post")
    def post_thread():
        async def post(text: str) -> str:
            """Start a new top-level thread for a distinct finding or question.

            Args:
                text: Post text, up to 2000 characters.
            """
            return invoke("board_post", board.post, text)
        return post

    @tool(name="board_reply")
    def reply_thread():
        async def reply(parent_id: int, text: str,
                        relation: Literal["agree", "disagree", "question", "comment"] = "comment") -> str:
            """Reply to a post or reply you have read, creating a child in its thread.

            Args:
                parent_id: ID of the message being answered (must already have been read).
                text: Reply text, up to 2000 characters; explain your response.
                relation: Explicit stance: agree, disagree, question, or comment.
            """
            return invoke("board_reply", board.post, text, parent_id=parent_id, relation=relation)
        return reply

    return [list_threads(), read_threads(), post_thread(), reply_thread()]


def _exception_status(error: BaseException) -> tuple[set[int], set[str]]:
    """Collect sanitized status attributes through common retry wrappers."""
    pending = [error]
    seen: set[int] = set()
    codes: set[int] = set()
    names: set[str] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for attr in ("status_code", "status", "code"):
            value = getattr(current, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                codes.add(value)
            elif isinstance(value, str):
                names.add(value.lower())
        for attr in ("__cause__", "__context__"):
            cause = getattr(current, attr, None)
            if isinstance(cause, BaseException):
                pending.append(cause)
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                nested = last_attempt.exception()
            except Exception:
                nested = None
            if isinstance(nested, BaseException):
                pending.append(nested)
    return codes, names


def swarm_adapter(inner_solver: Solver, agent_id: str, board: SharedBoard,
                  scheduler: FairScheduler,
                  stop_check: Callable[[str], str | None] | None = None,
                  ready_hook: Callable[[TaskState], Awaitable[None]] | None = None,
                  model_instance: Any = None,
                  input_hook: Callable[[str, list[Any]], list[Any]] | None = None) -> Solver:
    """Wrap an initialized Inspect solver with board tools and fair generation.

    Board access is enabled only when ``board.enabled`` is true. The model never
    supplies its author id: board posts use the trusted ``agent_id`` closure.
    ``input_hook`` may return an extended copy of each request's messages; the
    additions are sent to the model but never stored in the conversation.
    """
    if not callable(inner_solver):
        raise TypeError("inner_solver must be an initialized Inspect solver")
    if agent_id not in scheduler.agent_ids:
        raise ValueError("agent_id must be registered with the scheduler")

    @solver
    def adapted() -> Solver:
        async def run(state: TaskState, generate: Callable[..., Any]) -> TaskState:
            tools_added = False
            ready_done = False
            original_model_generate = model_instance.generate if model_instance is not None else None

            async def fair_generate(current_state: TaskState, *args: Any, **kwargs: Any) -> TaskState:
                nonlocal tools_added, ready_done
                if ready_hook is not None and not ready_done:
                    await ready_hook(current_state)
                    ready_done = True
                reason = await asyncio.to_thread(stop_check, agent_id) if stop_check is not None else None
                if reason:
                    await scheduler.finish(agent_id, reason)
                    current_state.completed = True
                    metadata = dict(getattr(current_state, "metadata", None) or {})
                    metadata["swarm_stop_reason"] = reason
                    current_state.metadata = metadata
                    return current_state
                if board.enabled and not tools_added:
                    # Upstream solvers may set their own tools before generation.
                    current_state.tools = list(getattr(current_state, "tools", None) or []) + _board_tools(board, agent_id, scheduler)
                    tools_added = True
                if not await scheduler.acquire(agent_id):
                    # Inspect's chain/basic-agent solvers stop cleanly when completed,
                    # preserving scoring of the best state reached so far.
                    current_state.completed = True
                    metadata = dict(getattr(current_state, "metadata", None) or {})
                    metadata["swarm_stop_reason"] = scheduler.snapshot()[agent_id]["status"]
                    current_state.metadata = metadata
                    return current_state
                reason = await asyncio.to_thread(stop_check, agent_id) if stop_check is not None else None
                if reason:
                    await scheduler.release(agent_id)
                    await scheduler.finish(agent_id, reason)
                    current_state.completed = True
                    metadata = dict(getattr(current_state, "metadata", None) or {})
                    metadata["swarm_stop_reason"] = reason
                    current_state.metadata = metadata
                    return current_state
                started = time.monotonic()
                try:
                    if original_model_generate is not None:
                        model_input = kwargs.pop("_swarm_model_input")
                        current_state.output = await original_model_generate(input=model_input, tools=current_state.tools, **kwargs)
                        return current_state
                    return await generate(current_state, *args, **kwargs)
                except BaseException as error:
                    codes, names = _exception_status(error)
                    reason = await asyncio.to_thread(stop_check, agent_id) if stop_check is not None else None
                    if 402 in codes and reason in (None, "budget", "budget_exhausted"):
                        reason = "budget"
                    elif 503 in codes and reason in ("accounting", "accounting_uncertain"):
                        reason = "accounting_uncertain"
                    if reason in ("budget", "budget_exhausted", "accounting", "accounting_uncertain", "dollars", "tokens", "deadline", "reservation_exceeded"):
                        await scheduler.finish(agent_id, "accounting_uncertain" if "accounting" in reason else reason)
                        current_state.completed = True
                        metadata = dict(getattr(current_state, "metadata", None) or {})
                        metadata["swarm_stop_reason"] = "accounting_uncertain" if "accounting" in reason else reason
                        current_state.metadata = metadata
                        return current_state
                    raise
                finally:
                    scheduler.record_api_time(agent_id, time.monotonic() - started)
                    await scheduler.release(agent_id)

            async def intercepted(input, tools=None, **kwargs):
                from inspect_ai.model import ModelOutput, ChatMessageAssistant, ChatCompletionChoice
                if input_hook is not None:
                    input = input_hook(agent_id, list(input))
                await fair_generate(state, _swarm_model_input=input, **kwargs)
                if state.completed:
                    return ModelOutput(model=str(model_instance), choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=""), stop_reason="stop")])
                return state.output

            if model_instance is not None:
                model_instance.generate = intercepted
            try:
                # basic_agent directly invokes its model, bypassing Generate.
                result = await inner_solver(state, generate if model_instance is not None else fair_generate)
            except asyncio.CancelledError:
                await scheduler.finish(agent_id, "cancelled")
                raise
            except BaseException:
                await scheduler.finish(agent_id, "error")
                raise
            else:
                await scheduler.finish(agent_id, "finished")
                metadata = dict(getattr(result, "metadata", None) or {})
                metadata["swarm"] = scheduler.snapshot()[agent_id]
                result.metadata = metadata
                return result
            finally:
                if model_instance is not None:
                    model_instance.generate = original_model_generate

        return run

    return adapted()
