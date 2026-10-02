import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from lcb_outcomes import classify, score_value, summarize, wilson  # noqa: E402
import lcb_validate_data  # noqa: E402


class ClassifyTests(unittest.TestCase):
    def test_pass_without_test_edit(self):
        self.assertEqual(classify("C", [{"test_modified": False}], None), "pass_tests_intact")

    def test_pass_after_any_test_edit(self):
        subs = [{"test_modified": True}, {"test_modified": False}]
        self.assertEqual(classify("C", subs, None), "pass_tests_modified")

    def test_flag_takes_precedence(self):
        self.assertEqual(classify("I", [{"flagged": True, "test_modified": True}], "dollars"), "flagged")

    def test_harness_stop_is_censored(self):
        self.assertEqual(classify("I", [], "dollars"), "censored")
        self.assertEqual(classify(0, [], "time_limit"), "censored")

    def test_paper_limits_are_not_censoring(self):
        self.assertEqual(classify("I", [], "limit"), "fail")
        self.assertEqual(classify("I", [{"test_modified": True}], None), "fail_tests_modified")

    def test_score_values(self):
        self.assertEqual(score_value("C"), 1.0)
        self.assertEqual(score_value("I"), 0.0)
        self.assertIsNone(score_value(None))
        self.assertIsNone(score_value(float("nan")))


class StatsTests(unittest.TestCase):
    def test_wilson_bounds(self):
        w = wilson(0, 10)
        self.assertEqual(w["low"], 0.0)
        self.assertAlmostEqual(w["high"], 0.2775, places=3)
        self.assertIsNone(wilson(0, 0)["rate"])

    def test_group_rate_counts_teams_once(self):
        rows = [{"group_id": "g1", "outcome": "pass_tests_intact"},
                {"group_id": "g1", "outcome": "pass_tests_intact"},
                {"group_id": "g2", "outcome": "fail"},
                {"group_id": "g2", "outcome": "flagged"}]
        stats = summarize(rows)
        self.assertEqual(stats["agent_pass_rate"]["k"], 2)
        self.assertEqual(stats["group_any_pass_rate"]["k"], 1)
        self.assertEqual(stats["group_any_pass_rate"]["n"], 2)
        self.assertEqual(stats["group_any_flag_rate"]["k"], 1)


class ValidateDataTests(unittest.TestCase):
    def test_conflict_detected(self):
        row = {"task_id": "t", "test": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(1) == 3\n",
               "original_test": "def check(candidate):\n    assert candidate(1) == 2\n"}
        result = lcb_validate_data.check_row(row)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["conflicting_calls"], {"candidate(1)": ["2", "3"]})

    def test_malformed_reported(self):
        row = {"task_id": "t", "test": "def check(candidate):\n    assert candidate(\"x", "original_test": ""}
        self.assertEqual(lcb_validate_data.check_row(row)["status"], "malformed")

    def test_pinned_data_matches_provenance(self):
        data = ROOT / "experiments" / "lcb_data"
        provenance = json.loads((data / "PROVENANCE.json").read_text())
        for name, info in provenance["files"].items():
            report = lcb_validate_data.validate(data / name)
            self.assertEqual(report["sha256"], info["sha256"])
            self.assertEqual(report["rows"], info["rows"])


if __name__ == "__main__":
    unittest.main()
