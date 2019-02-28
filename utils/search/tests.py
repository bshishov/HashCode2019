import unittest
from search.search import SortedQueue


class SortedQueueTest(unittest.TestCase):
    def test_sorted_integer_queue(self):
        sq = SortedQueue()
        sq.put(4)
        sq.put(1)
        sq.put(2)
        self.assertEqual(sq.qsize(), 3)

        self.assertEqual(sq.get(), 1)
        self.assertEqual(sq.qsize(), 2)

        self.assertEqual(sq.get(), 2)
        self.assertEqual(sq.qsize(), 1)

        self.assertEqual(sq.get(), 4)
        self.assertEqual(sq.qsize(), 0)
