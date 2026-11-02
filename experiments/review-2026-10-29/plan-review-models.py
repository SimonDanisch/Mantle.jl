"""Small counterexamples to plan sketches, not tests of the Mantle runtime."""

import json
from pathlib import Path


def reachable(edges, source, target):
    pending, visited = [source], set()
    while pending:
        node = pending.pop()
        if node == target:
            return True
        if node not in visited:
            visited.add(node)
            pending.extend(b for a, b in edges if a == node)
    return False


# Both readers use A, on independent queues. The last reader in a
# topological listing need not complete after the other reader.
edges = [("produce_A", "slow_read_A"), ("produce_A", "fast_read_A")]
proposed = edges + [("fast_read_A", "overwrite_as_B")]
assert not reachable(proposed, "slow_read_A", "overwrite_as_B")
slow_interval, overwrite_interval = (1, 101), (2, 3)
assert max(slow_interval[0], overwrite_interval[0]) < min(
    slow_interval[1], overwrite_interval[1]
)
fixed = proposed + [("slow_read_A", "overwrite_as_B")]
assert all(reachable(fixed, reader, "overwrite_as_B")
           for reader in ("slow_read_A", "fast_read_A"))

# A queue's new submission can spend its entire host wait waiting on
# several legal submissions of another queue. No execution exceeds timeout.
jobtimeout = 1.0
dependency_segments = [0.75, 0.75, 0.75]
assert all(duration < jobtimeout for duration in dependency_segments)
assert sum(dependency_segments) > 2 * jobtimeout

# Preserving a shadow map on a dirty frame leaves the depth of a removed
# occluder. Conditional draws cannot erase it without a clear/reset.
old_depth = [0.2, 1.0]
redrawn_without_clear = old_depth.copy()  # occluder was removed
correct_depth = [1.0, 1.0]
assert redrawn_without_clear != correct_depth

# A length-writing launch inserts a PrepareNode into its group. The
# documented loopable methods reject that automatically inserted node.
loopable_kinds = {"KernelNode", "TraceNode"}
generated_body = ["KernelNode", "PrepareNode"]
assert not all(kind in loopable_kinds for kind in generated_body)

result = {
    "scope": "Abstract counterexamples only; no Julia or GPU execution",
    "alias_frontier_counterexample": True,
    "dependency_wait_exceeds_watchdog_without_long_kernel": True,
    "dirty_shadow_without_clear_keeps_removed_occluder": True,
    "length_prepare_rejected_by_documented_loopable_methods": True,
}
Path(__file__).with_name("plan-review-model-results.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result, indent=2))
