# Helpers shared between HW-HWTLAS test files (test_hwtlas_stress.jl,
# test_hwtlas_mesh_update.jl, …).  Each test includes this with an
# `isdefined` guard so re-includes in the same `Main` are no-ops; the
# files still work standalone. Not called `translation`: Makie exports that name,
# so in a `Main` that had loaded Makie the guard found Makie's and skipped this
# file, and every call went to `Makie.translation(::Transformable)`.

using StaticArrays: SMatrix

"""Translation as a `Mat4f` (SMatrix{4,4,Float32,16})."""
tlastranslation(dx, dy, dz) = SMatrix{4,4,Float32,16}(
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    dx, dy, dz, 1,
)
