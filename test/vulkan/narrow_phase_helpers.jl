# Helpers shared between narrow-phase test files (test_gjk.jl, test_epa.jl,
# test_narrow_phase_kernel.jl, test_narrow_phase_contacts.jl).  These tests
# all run in `Main` via the top-level test runner, so we previously had
# four files each defining the same `translation_transform` / `tx` /
# `rotation_z_transform` and the test runner would print a "Method
# definition ... overwritten at ..." warning for every duplicate.
#
# Each test file includes this once with an `isdefined` guard, so the
# helpers exist exactly once per runtests session and the files still
# work standalone.

function translation_transform(x, y, z)
    (1f0, 0f0, 0f0, Float32(x),
     0f0, 1f0, 0f0, Float32(y),
     0f0, 0f0, 1f0, Float32(z))
end

function rotation_z_transform(angle, tx=0f0, ty=0f0, tz=0f0)
    c = Float32(cos(angle))
    s = Float32(sin(angle))
    (c,    -s,   0f0,  Float32(tx),
     s,     c,   0f0,  Float32(ty),
     0f0,  0f0,  1f0,  Float32(tz))
end

# Identity transform (no rotation, no translation).
const ID = translation_transform(0, 0, 0)

# Shorter alias used by `test_narrow_phase_{kernel,contacts}.jl`.
const tx = translation_transform
