import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import index_runs  # noqa: E402


def make_run(base: Path, name: str, started: str, *, mode="live", summary=None, accounting=None) -> Path:
    run = base / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"started_at": started, "mode": mode, "condition": "board",
                                                   "agents_per_group": 2, "tasks": ["lcbhard_0"]}))
    if summary is not None:
        (run / "summary.json").write_text(json.dumps(summary))
    if accounting is not None:
        (run / "billing").mkdir()
        (run / "billing" / "accounting.json").write_text(json.dumps(accounting))
    return run


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        (self.root / "experiments").mkdir()
        self.patches = [patch.object(index_runs, "ROOT", self.root), patch.object(index_runs, "RUNS", self.runs),
                        patch.object(index_runs, "ANNOTATIONS", self.root / "experiments" / "run_annotations.json")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_orders_by_manifest_start_not_file_time(self):
        done = {"status": "completed", "stats": {"outcome_counts": {"fail": 2}}, "rows": [{}, {}]}
        later = make_run(self.runs, "b-later", "2026-10-03T02:00:00+00:00", summary=done)
        earlier = make_run(self.runs, "a-earlier", "2026-10-03T01:00:00+00:00", summary=done)
        os.utime(earlier / "summary.json", (2_000_000_000, 2_000_000_000))  # Touched most recently.
        index_runs.write_index()
        runs = json.loads((self.runs / "index.json").read_text())["runs"]
        self.assertEqual([r["run_id"] for r in runs], ["a-earlier", "b-later"])
        self.assertEqual([r["live_run_number"] for r in runs], [1, 2])

    def test_validity_rules_and_annotation_override(self):
        make_run(self.runs, "aborted", "2026-10-03T01:00:00+00:00",
                 accounting={"spent_usd": 0.07, "reserved_usd": 0.11})
        make_run(self.runs, "censored", "2026-10-03T02:00:00+00:00",
                 summary={"status": "completed", "stats": {"outcome_counts": {"censored": 1}}, "rows": [{}]})
        make_run(self.runs / "checks", "check", "2026-10-03T03:00:00+00:00", mode="check",
                 summary={"status": "completed", "rows": []})
        make_run(self.runs, "overridden", "2026-10-03T04:00:00+00:00",
                 summary={"status": "completed", "stats": {"outcome_counts": {"fail": 1}}, "rows": [{}]})
        (self.root / "experiments" / "run_annotations.json").write_text(
            json.dumps({"runs": {"overridden": {"validity": "censored", "caveats": ["timeout"]}}}))
        index_runs.write_index()
        runs = {r["run_id"]: r for r in json.loads((self.runs / "index.json").read_text())["runs"]}
        self.assertEqual(runs["aborted"]["validity"], "aborted")
        self.assertEqual(runs["aborted"]["spend"], {"recorded_usd": 0.07, "possibly_unrecorded_usd": 0.11,
                                                    "accounting_uncertain": False})
        self.assertEqual(runs["censored"]["validity"], "censored")
        self.assertEqual(runs["check"]["validity"], "test")
        self.assertEqual(runs["overridden"]["validity"], "censored")
        self.assertEqual(runs["overridden"]["prompt"], "D")  # Runs before --prompt existed used D.
        self.assertIn("timeout", (self.runs / "INDEX.md").read_text())


if __name__ == "__main__":
    unittest.main()
