"""Rank distinct, non-overlapping implementation pairs from semantic vectors."""
from collections import Counter

import numpy as np


def owner(chunk):
    return (chunk["file"], chunk.get("owner_start", chunk["start"]),
            chunk.get("owner_end", chunk["end"]))


def rank_pairs(chunks, vectors, *, same_file=False, limit=100, per_file_pair=None, min_lines=1):
    if limit <= 0:
        return []
    substantive = [sum(bool(line.strip()) for line in c.get("code", "").splitlines()) >= min_lines
                   for c in chunks]
    similarities = vectors @ vectors.T
    left, right = np.triu_indices(len(chunks), k=1)
    order = np.argsort(-similarities[left, right], kind="stable")
    seen, counts, result = set(), Counter(), []
    for index in order:
        a, b = int(left[index]), int(right[index])
        ca, cb = chunks[a], chunks[b]
        if min_lines > 1 and not (substantive[a] and substantive[b]):
            continue
        if (ca["file"] == cb["file"]) != same_file:
            continue
        oa, ob = owner(ca), owner(cb)
        # A method, its windows and nested methods are not independent copies.
        if same_file and max(oa[1], ob[1]) <= min(oa[2], ob[2]):
            continue
        pair = tuple(sorted((oa, ob)))
        files = tuple(sorted((ca["file"], cb["file"])))
        if pair in seen or (per_file_pair is not None and counts[files] >= per_file_pair):
            continue
        seen.add(pair)
        counts[files] += 1
        result.append((float(similarities[a, b]), a, b))
        if len(result) >= limit:
            break
    return result
