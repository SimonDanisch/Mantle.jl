import unittest

import numpy as np

from candidates import rank_pairs


def chunk(file, start, end, *, owner_start=None, owner_end=None):
    return {"file": file, "start": start, "end": end,
            "owner_start": start if owner_start is None else owner_start,
            "owner_end": end if owner_end is None else owner_end}


class CandidateDiscovery(unittest.TestCase):
    def test_same_file_methods_are_discoverable(self):
        chunks = [chunk("a", 1, 10), chunk("a", 20, 30), chunk("b", 1, 10)]
        vectors = np.array([[1., 0.], [1., 0.], [0., 1.]])
        self.assertEqual(rank_pairs(chunks, vectors, same_file=True), [(1., 0, 1)])

    def test_windows_do_not_duplicate_their_method_pair(self):
        chunks = [chunk("a", 1, 100), chunk("a", 1, 40, owner_end=100),
                  chunk("b", 1, 100), chunk("b", 25, 64, owner_start=1, owner_end=100)]
        self.assertEqual(len(rank_pairs(chunks, np.ones((4, 1)))), 1)

    def test_nested_methods_are_not_clones_of_their_parent(self):
        chunks = [chunk("a", 1, 100), chunk("a", 20, 40), chunk("a", 110, 120)]
        pairs = rank_pairs(chunks, np.ones((3, 1)), same_file=True)
        self.assertEqual({(a, b) for _, a, b in pairs}, {(0, 2), (1, 2)})

    def test_diversity_exposes_other_file_pairs(self):
        chunks = [chunk("a", 1, 10), chunk("a", 20, 30), chunk("b", 1, 10),
                  chunk("b", 20, 30), chunk("c", 1, 10)]
        pairs = rank_pairs(chunks, np.ones((5, 1)), limit=3, per_file_pair=1)
        files = {tuple(sorted((chunks[a]["file"], chunks[b]["file"]))) for _, a, b in pairs}
        self.assertEqual(files, {("a", "b"), ("a", "c"), ("b", "c")})

    def test_thin_wrappers_do_not_hide_larger_methods(self):
        chunks = [chunk("a", 1, 1), chunk("a", 10, 20), chunk("a", 30, 40)]
        for c, n in zip(chunks, (1, 8, 8)):
            c["code"] = "x\n" * n
        pairs = rank_pairs(chunks, np.ones((3, 1)), same_file=True, min_lines=6)
        self.assertEqual(pairs, [(1., 1, 2)])

    def test_all_pairs_search_survives_many_self_windows(self):
        chunks = [chunk("a", i, i+1, owner_start=1, owner_end=100) for i in range(1, 41)]
        chunks.append(chunk("b", 1, 10))
        self.assertEqual(len(rank_pairs(chunks, np.ones((41, 1)))), 1)


if __name__ == "__main__":
    unittest.main()
