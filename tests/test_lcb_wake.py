"""Offline tests for the wake-condition fields of the LCB campaign."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.lcb_swarm_campaign import wake_fields


class WakeFieldsTests(unittest.TestCase):
    def test_end_record_is_read_for_agents_stopped_by_a_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Path(directory)
            # Never went idle (no idle_scores), stopped by the message limit with one unanswered post.
            (agent / "wake.json").write_text(json.dumps({"wakes": 0, "episode_end": None, "unread_at_end": 1}))
            fields = wake_fields(agent, [], "fail")
            self.assertEqual(fields["outcome_first_finish"], "fail")
            self.assertIsNone(fields["episode_end"])
            self.assertEqual(fields["unread_at_end"], 1)

    def test_missing_record_gives_none(self):
        with tempfile.TemporaryDirectory() as directory:
            fields = wake_fields(Path(directory), [], "pass_tests_intact")
            self.assertIsNone(fields["unread_at_end"])
            self.assertEqual(fields["wakes"], 0)


if __name__ == "__main__":
    unittest.main()
