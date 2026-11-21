# The backend a device test runs on. `runtests.jl` includes each such file once per
# usable backend (`foreachbackend`), which sets `Main.MANTLE_TEST_BACKEND`; a bare
# `include` from the REPL gets the default one. Nothing in a test names a backend.
import Mantle
const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ? Main.MANTLE_TEST_BACKEND :
                                                           Mantle.defaultbackend()

# The element types among `Ts` this backend computes in: a Float64 one, real or
# complex, only where the device has Float64.
testeltypes(Ts...) = filter(T -> Mantle.supports_float64(TESTBACKEND) || real(T) !== Float64, Ts)

# Whether the device runs the cooperative-matrix kernels this build's staged GEMM
# is written at: it has cooperative matrices, at the tile those kernels are
# emitted for. `Mantle.staged_gemm_tile` documents the comparison.
function hasstagedgemm()
    c = Mantle.caps(TESTBACKEND)
    return c.coopmat && (t = Mantle.staged_gemm_tile()) !== nothing && c.tile == t
end

# Two guarantees Metal.jl does not give yet, each the subject of a performance PR of its
# own (the Metal.jl fork Mantle used before had both): submitting recorded runs and
# launches without allocating, and finding a kernel compiled in an earlier process. Their
# tests are marked broken on Metal until those land, and then report an unexpected pass.
metalperfpending() = nameof(typeof(Mantle.Device(TESTBACKEND))) === :MetalDevice
