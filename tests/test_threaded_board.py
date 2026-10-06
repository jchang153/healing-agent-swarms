"""Thread topology, exposure provenance and read accounting, without model calls."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.swarm_tools import SharedBoard, FairScheduler, _board_tools


class ThreadedBoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "board.jsonl"
        self.board = SharedBoard("g", True, self.log, structure="threaded")

    def tearDown(self):
        self.tmp.cleanup()

    def test_nested_replies_and_thread_scoped_reads(self):
        root = self.board.post("a", "finding")
        other = self.board.post("c", "separate finding")
        self.board.read_threads("b", root["id"])
        reply = self.board.post("b", "confirmed", parent_id=root["id"], relation="agree")
        self.board.read_threads("c", root["id"])
        child = self.board.post("c", "why?", parent_id=reply["id"], relation="question")
        tree = self.board.read_threads("reader", root["id"])[0]
        self.assertEqual(tree["replies"][0]["replies"][0]["id"], child["id"])
        self.assertEqual(child["thread_id"], root["id"])
        self.assertEqual([m["id"] for m in self.board.unread("reader")], [other["id"]])
        tree["replies"].clear()
        self.assertEqual(len(self.board.read_threads("reader", 0)[0]["replies"]), 1)

    def test_listing_does_not_deliver_bodies_and_default_reads_clear_unread(self):
        self.board.post("a", "private message body")
        listed = self.board.list_threads("b")
        self.assertNotIn("private message body", json.dumps(listed))
        self.assertEqual(listed[0]["unread_message_ids"], [0])
        self.assertEqual(len(self.board.unread("b")), 1)
        self.board.read_threads("b")
        self.assertEqual(self.board.read_threads("b"), [])
        self.board.post("a", "follow-up", parent_id=0)
        trees = self.board.read_threads("b")
        self.assertEqual(trees[0]["id"], 0)
        self.assertEqual(trees[0]["replies"][0]["id"], 1)
        self.assertEqual(self.board.unread("b"), [])
        events = [json.loads(l) for l in self.log.read_text().splitlines()]
        self.assertEqual(events[1]["exposed_message_ids"], [])
        self.assertEqual(events[-1]["exposed_message_ids"], [0, 1])

    def test_invalid_or_unseen_parents_cannot_create_replies(self):
        self.board.post("a", "root")
        for parent in (True, -1, 99, "0", 0):
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                self.board.post("b", "reply", parent_id=parent)
        self.board.read_threads("b", 0)
        with self.assertRaises(ValueError):
            self.board.post("b", "reply", parent_id=0, relation="invalid")
        with self.assertRaises(ValueError):
            self.board.post("b", "new", relation="agree")
        reply = self.board.post("b", "okay", parent_id=0)
        for thread in (True, -1, 99, "0", reply["id"]):
            with self.subTest(thread=thread), self.assertRaises(ValueError):
                self.board.read_threads("b", thread)
        self.assertEqual(reply["id"], 1)  # Rejections consume no IDs.

    def test_new_root_is_separate_from_existing_thread(self):
        self.board.post("a", "first")
        self.board.read_threads("b")
        second = self.board.post("b", "distinct topic")
        self.assertIsNone(second["parent_id"])
        self.assertEqual(second["thread_id"], second["id"])
        self.assertNotIn("read_before_post_ids", second)
        event = json.loads(self.log.read_text().splitlines()[-1])
        self.assertEqual(event["read_before_post_ids"], [0])
        self.assertNotIn("read_before_post_ids", event["message"])

    def test_limits_and_group_isolation(self):
        other = SharedBoard("other", True, Path(self.tmp.name) / "other.jsonl",
                            structure="threaded", max_messages=1)
        other.post("a", "other group")
        self.assertEqual(self.board.list_threads("b"), [])
        with self.assertRaises(ValueError):
            other.post("a", "reply", parent_id=0)
        disabled = SharedBoard("off", False, self.log, structure="threaded")
        with self.assertRaises(RuntimeError):
            disabled.read_threads("b")

    def test_flat_contract_remains_unchanged(self):
        board = SharedBoard("flat", True, self.log)
        self.assertEqual(set(board.post("a", "old format")), {"id", "author", "text", "timestamp"})
        with self.assertRaises(ValueError):
            board.post("a", "reply", parent_id=0)


class ThreadedToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspect_tools_route_tree_reads_and_explicit_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            board = SharedBoard("g", True, Path(d) / "board.jsonl", structure="threaded")
            scheduler = FairScheduler(["a", "b"], 10, 30)
            a = _board_tools(board, "a", scheduler)
            b = _board_tools(board, "b", scheduler)
            self.assertEqual(len(a), 4)
            root = json.loads(await a[2](text="finding"))
            await b[0]()
            await b[1](thread_id=root["id"])
            reply = json.loads(await b[3](parent_id=0, text="verified", relation="agree"))
            self.assertEqual(reply["author"], "b")
            self.assertEqual(reply["relation"], "agree")
            self.assertEqual(json.loads(await a[1]())[0]["replies"][0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
