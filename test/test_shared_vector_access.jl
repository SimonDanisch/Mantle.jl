using Test, Mantle
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

# Shared memory declared as scalars and accessed four at a time, the way Bonsai's
# prefill attention reads its staged queries (`reinterpret(NTuple{4,VecElement{Float32}}, qs)`).
# On Vulkan this failed twice: `reinterpret` fell to Base's `ReinterpretArray`, whose
# field-offset `ccall` no kernel can run, and once Lava had its own, the access came
# out as an `OpPtrAccessChain` to a `v4float` off a `float` array, which `spirv-val`
# refuses (a logical pointer cannot be retyped). Lava now splits each such vector
# access into one access per component. Checked by value: an access chain that
# validates can still land on the wrong element.

const SV_WG = 64
const SV4 = NTuple{4, VecElement{Float32}}

# Stage scalars, read them back as vectors: each of the first 16 work-items one
# vector at a dynamic index, and the first also the vector at a constant one.
function shared_vector_read(out, outc, x)
    sh = KI.localmemory(Float32, Val((SV_WG,)), Val(1))
    t = KI.get_local_id().x
    @inbounds sh[t] = x[t]
    KI.barrier()
    sh4 = reinterpret(SV4, sh)
    if t <= SV_WG ÷ 4
        v = @inbounds sh4[t]
        i = 4 * (t - 1)
        @inbounds out[i + 1] = v[1].value
        @inbounds out[i + 2] = v[2].value
        @inbounds out[i + 3] = v[3].value
        @inbounds out[i + 4] = v[4].value
    end
    if t == 1
        c = @inbounds sh4[2]
        @inbounds outc[1] = c[1].value
        @inbounds outc[2] = c[2].value
        @inbounds outc[3] = c[3].value
        @inbounds outc[4] = c[4].value
    end
    return nothing
end

# Write vectors of Float32 into UInt32 memory, read it back as scalars: the
# components are stored as the memory's own type.
function shared_vector_write(out, x)
    sh = KI.localmemory(UInt32, Val((SV_WG,)), Val(1))
    t = KI.get_local_id().x
    if t <= SV_WG ÷ 4
        i = 4 * (t - 1)
        v = (VecElement(@inbounds x[i + 1]), VecElement(@inbounds x[i + 2]),
             VecElement(@inbounds x[i + 3]), VecElement(@inbounds x[i + 4]))
        @inbounds reinterpret(SV4, sh)[t] = v
    end
    KI.barrier()
    @inbounds out[t] = reinterpret(Float32, sh[t])
    return nothing
end

@testset "shared scalars accessed as 4-vectors" begin
    backend = TESTBACKEND
    x = Float32.(1:SV_WG) .+ 0.25f0
    xd = Mantle.devicearray(backend, x)

    out = Mantle.devicearray(backend, fill(-1f0, SV_WG))
    outc = Mantle.devicearray(backend, fill(-1f0, 4))
    KI.@launch backend numgroups = 1 workgroupsize = SV_WG shared_vector_read(out, outc, xd)
    KI.synchronize(backend)
    @test Array(out) == x
    @test Array(outc) == x[5:8]

    out = Mantle.devicearray(backend, fill(-1f0, SV_WG))
    KI.@launch backend numgroups = 1 workgroupsize = SV_WG shared_vector_write(out, xd)
    KI.synchronize(backend)
    @test Array(out) == x
end
