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
from typing import Any, Awaitable, Callable

from inspect_ai.solver import TaskState, Solver, solver
from inspect_ai.tool import tool


@dataclass(frozen=True)
class _Message:
    id: int
    author: str
    text: str
    timestamp: str


class SharedBoard:
    """A bounded board whose authors are supplied by the trusted caller."""

    def __init__(self, group_id: str, enabled: bool, log_path: str | Path,
                 *, max_messages: int = 1000, max_text_length: int = 2000) -> None:
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages < 1:
            raise ValueError("max_messages must be a positive integer")
        if isinstance(max_text_length, bool) or not isinstance(max_text_length, int) or max_text_length < 1:
            raise ValueError("max_text_length must be a positive integer")
        self.group_id = group_id
        self.enabled = enabled
        self.log_path = Path(log_path)
        self.max_messages = max_messages
        self.max_text_length = max_text_length
        self._messages: list[_Message] = []
        self._lock = RLock()

    def _log(self, event: str, **fields: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(), group_id=self.group_id,
                   event=event, **fields)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def post(self, agent_id: str, text: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(text, str) or len(text) > self.max_text_length:
            raise ValueError(f"text must be a string up to {self.max_text_length} characters")
        with self._lock:
            if len(self._messages) >= self.max_messages:
                raise ValueError("Board capacity reached")
            item = _Message(id=len(self._messages), author=agent_id, text=text,
                            timestamp=datetime.now(timezone.utc).isoformat())
            self._messages.append(item)
            result = asdict(item)
            self._log("board_post", message=result)
            return result

    def read(self, agent_id: str, since: int = 0) -> list[dict[str, Any]]:
        if not self.enabled:
            raise RuntimeError("The shared board is disabled")
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if isinstance(since, bool) or not isinstance(since, int) or since < 0:
            raise ValueError("since must be a non-negative integer message id")
        with self._lock:
            result = [asdict(item) for item in self._messages if item.id >= since]
            self._log("board_read", agent=agent_id, since=since,
                      exposed_message_ids=[item["id"] for item in result])
            return result


class FairScheduler:
    """FIFO scheduler allowing one complete generate call at a time."""

    def __init__(self, agent_ids: list[str] | tuple[str, ...], max_turns: int,
                 seconds: float) -> None:
        ids = tuple(agent_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("agent_ids must contain unique, non-empty strings")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        self.agent_ids = ids
        self.max_turns = max_turns
        self.seconds = float(seconds)
        self._deadline: float | None = None
        self._owner: str | None = None
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
            if agent_id not in self._queue and self._owner != agent_id:
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
                if self._owner is None and self._queue and self._queue[0] == agent_id:
                    self._queue.popleft()
                    self._owner = agent_id
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
            if self._owner == agent_id:
                self._owner = None
            if self._status.get(agent_id) == "running":
                self._status[agent_id] = "waiting"
            self._condition.notify_all()

    async def finish(self, agent_id: str, status: str = "finished") -> None:
        if agent_id not in self._turns:
            return
        async with self._condition:
            self._remove_waiter(agent_id)
            if self._owner == agent_id:
                self._owner = None
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
                  model_instance: Any = None) -> Solver:
    """Wrap an initialized Inspect solver with board tools and fair generation.

    Board access is enabled only when ``board.enabled`` is true. The model never
    supplies its author id: board posts use the trusted ``agent_id`` closure.
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
