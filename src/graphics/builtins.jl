# DELETED in phase 1.4: see docs/mantle-owns-it.md
#
# This file declared `vertex_index`, `instance_index`, the `frag_coord` family,
# `SHADER_BUILTINS` and `clip_y`, with a host method per name that errored.
#
# It is the wrong package. Lava does not depend on Mantle — Mantle depends on
# Lava — so Lava cannot override `Mantle.vertex_index`, and the bridge
# `Mantle.$f() = Lava.$f()` in `vulkan/graphics/api.jl` existed only to make up
# for that. KernelInterface is below both and already declares the compute half
# (`get_global_id`, `barrier`, `sub_group_reduce_add`); phase 2.1 puts the
# graphics and ray-tracing halves beside them, where Lava overrides directly
# and no bridge is needed.
