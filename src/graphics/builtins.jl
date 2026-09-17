#
# `vertex_index`, `instance_index`, the `frag_coord` family, `SHADER_BUILTINS`
# and `clip_y` are NOT declared here.
#
# Mantle is the wrong package for them. Lava does not depend on Mantle, Mantle
# depends on Lava, so Lava cannot override `Mantle.vertex_index` and a bridge
# `Mantle.f() = Lava.f()` would be needed for every name. KernelInterface is
# below both and already declares the compute half (`get_global_id`, `barrier`,
# `sub_group_reduce_add`), so the graphics and ray-tracing halves sit beside
# them, where Lava overrides directly.
