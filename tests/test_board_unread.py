"""Per-agent unread tracking and per-turn board notices."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from swarm_tools import SharedBoard  # noqa: E402


class UnreadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "board.jsonl"
        self.board = SharedBoard("g", True, self.log)

    def tearDown(self):
        self.tmp.cleanup()

    def test_own_posts_are_never_unread(self):
        self.board.post("a", "hello")
        self.assertEqual(self.board.unread("a"), [])
        self.assertEqual([m["id"] for m in self.board.unread("b")], [0])

    def test_read_clears_only_delivered_messages(self):
        self.board.post("a", "one")
        self.board.post("c", "two")
        self.board.read("b", since=1)
        self.assertEqual([m["id"] for m in self.board.unread("b")], [0])
        self.board.read("b", since=0)
        self.assertEqual(self.board.unread("b"), [])
        self.board.post("a", "three")
        self.assertEqual([m["id"] for m in self.board.unread("b")], [2])

    def test_notice_text_and_log(self):
        self.assertIn("No unread messages", self.board.notice("b"))
        self.board.post("a", "x")
        self.board.post("c", "y")
        text = self.board.notice("b")
        self.assertIn("2 unread messages", text)
        self.assertIn("from a, c", text)
        events = [json.loads(line) for line in self.log.read_text().splitlines()]
        notices = [e for e in events if e["event"] == "board_notice"]
        self.assertEqual([n["unread_message_ids"] for n in notices], [[], [0, 1]])

    def test_disabled_board_has_no_notice(self):
        with self.assertRaises(RuntimeError):
            SharedBoard("g", False, self.log).notice("b")


if __name__ == "__main__":
    unittest.main()
