# Helpers shared between HW-TLAS test files (test_hwtlas_stress.jl,
# test_hwtlas_mesh_update.jl, …).  Each test includes this with an
# `isdefined` guard so re-includes in the same `Main` are no-ops; the
# files still work standalone.

using StaticArrays: SMatrix

"""Translation as a `Mat4f` (SMatrix{4,4,Float32,16})."""
translation(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)
