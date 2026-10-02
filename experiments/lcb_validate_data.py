"""Check that pinned Conflicting-LiveCodeBench tasks are mechanically impossible.

A task is confirmed impossible when its mutated ``check`` asserts the same call
(identical argument source) against two different expected values. Tasks whose
asserts do not follow the ``candidate(...) == expected`` form are reported as
unconfirmed for manual review rather than silently accepted, and tests that do
not parse are reported as malformed. The check also
verifies that each mutated test differs from its ``original_test``.

Deterministic and offline; no model calls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "lcb_data"


def _equality_pairs(test_source: str) -> list[tuple[str, str]]:
    """Return (call source, expected source) for ``assert candidate(...) == x``."""
    pairs = []
    for node in ast.walk(ast.parse(test_source)):
        if not isinstance(node, ast.Assert):
            continue
        cmp = node.test
        if not (isinstance(cmp, ast.Compare) and len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq)):
            continue
        left, right = cmp.left, cmp.comparators[0]
        for call, expected in ((left, right), (right, left)):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "candidate":
                pairs.append((ast.unparse(call), ast.unparse(expected)))
                break
    return pairs


def check_row(row: dict) -> dict:
    try:
        pairs = _equality_pairs(row["test"])
    except SyntaxError as error:
        # A test file that cannot parse fails for every submission, so it does
        # not measure test exploitation. Report it instead of counting it.
        return {"task_id": row["task_id"], "status": "malformed",
                "differs_from_original": row["test"].strip() != row["original_test"].strip(),
                "error": f"SyntaxError at line {error.lineno}", "equality_asserts": 0,
                "conflicting_calls": {}}
    seen: dict[str, set[str]] = defaultdict(set)
    for call, expected in pairs:
        seen[call].add(expected)
    conflicts = {call: sorted(values) for call, values in seen.items() if len(values) > 1}
    status = "confirmed" if conflicts else "unconfirmed"
    return {"task_id": row["task_id"], "status": status,
            "differs_from_original": row["test"].strip() != row["original_test"].strip(),
            "equality_asserts": len(pairs), "conflicting_calls": conflicts}


def validate(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    results = [check_row(r) for r in rows]
    ids = [r["task_id"] for r in rows]
    return {"file": path.name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rows": len(rows), "unique_task_ids": len(set(ids)),
            "impossible_type_values": sorted({r.get("impossible_type") for r in rows}),
            "confirmed": sum(r["status"] == "confirmed" for r in results),
            "unconfirmed": [r["task_id"] for r in results if r["status"] == "unconfirmed"],
            "malformed": [r["task_id"] for r in results if r["status"] == "malformed"],
            "unchanged_from_original": [r["task_id"] for r in results if not r["differs_from_original"]],
            "tasks": results}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=Path, default=DATA / "conflicting.jsonl")
    p.add_argument("--out", type=Path, default=DATA / "conflicting-validation.json")
    args = p.parse_args()
    report = validate(args.file)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))
    ok = (report["rows"] == report["unique_task_ids"] and not report["unchanged_from_original"]
          and report["impossible_type_values"] == ["conflicting"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
