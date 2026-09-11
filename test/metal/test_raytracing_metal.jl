"""
Hardware ray tracing on Metal: acceleration structures built on the device.

The first testset is the important one. `Mat3x4f` and `MTLPackedFloat4x3` are
both 48 bytes and both twelve `Float32`, and they are TRANSPOSES of each other —
reinterpreting one as the other type-checks, runs, and silently shears every
instance in the scene. Nothing downstream would report it; geometry would just
be in the wrong place.
"""

using Test, Mantle, Metal
const MTL = Metal.MTL

@testset "Metal: the transform is converted, not reinterpreted" begin
    id = Mantle.identity_transform()
    packed = Mantle.packedtransform(id)

    # Mantle/Vulkan: a row-major 3×4 — three rows of four.
    @test collect(reinterpret(Float32, [id])) ==
          Float32[1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0]
    # Metal: four columns of three. Same twelve floats, other order.
    @test collect(reinterpret(Float32, [packed])) ==
          Float32[1, 0, 0,  0, 1, 0,  0, 0, 1,  0, 0, 0]

    # The two really do differ, so a reinterpret really would be wrong. This is
    # the assertion that would have caught the bug.
    @test collect(reinterpret(Float32, [id])) != collect(reinterpret(Float32, [packed]))

    # …and the translation column survives the transpose intact, which is the
    # part a sheared matrix gets subtly wrong rather than obviously.
    t = Mantle.Mat3x4f(1, 0, 0, 7,  0, 1, 0, 8,  0, 0, 1, 9)
    pf = collect(reinterpret(Float32, [Mantle.packedtransform(t)]))
    @test pf[10:12] == Float32[7, 8, 9]      # the fourth column IS the translation
end

@testset "Metal: the instance descriptor is not Vulkan's" begin
    # 64 bytes on both, and the leading 48 are the transform on both. After that
    # they diverge: Vulkan packs a 24-bit index with an 8-bit mask and ends with
    # a 64-bit device ADDRESS of the structure; Metal has separate 32-bit fields
    # and ends with an INDEX into `instancedAccelerationStructures`.
    #
    # That divergence is why `VulkanInstanceRecord` stayed a Vulkan type instead
    # of becoming shared, and why only `Mat3x4f` is common.
    T = MTL.MTLAccelerationStructureInstanceDescriptor
    @test sizeof(T) == 64
    @test isbitstype(T)
    @test fieldoffset(T, 1) == 0                       # transform first
    @test sizeof(Mantle.Mat3x4f) == 48                 # …and it is 48 bytes
    @test :accelerationStructureIndex in fieldnames(T) # an index, not an address
end

@testset "Metal: build a BLAS on the device" begin
    d = Mantle.Device(Mantle.MetalAPI())
    @test MTL.supports_raytracing(d.dev)

    verts = Float32[0, 0, 0,  1, 0, 0,  0, 1, 0]
    vbuf = MTL.MTLBuffer(d.dev, sizeof(verts); storage = Metal.SharedStorage)
    unsafe_copyto!(convert(Ptr{Float32}, MTL.contents(vbuf)), pointer(verts), length(verts))

    blas = Mantle.build_blas(d, vbuf, 1)
    @test blas.handle.size > 0
end

@testset "Metal: build a HWTLAS over instances, then refit it" begin
    d = Mantle.Device(Mantle.MetalAPI())
    verts = Float32[0, 0, 0,  1, 0, 0,  0, 1, 0]
    vbuf = MTL.MTLBuffer(d.dev, sizeof(verts); storage = Metal.SharedStorage)
    unsafe_copyto!(convert(Ptr{Float32}, MTL.contents(vbuf)), pointer(verts), length(verts))
    blas = Mantle.build_blas(d, vbuf, 1)

    I = Mantle.identity_transform()
    shifted = Mantle.Mat3x4f(1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 3)
    tlas = Mantle.build_accel!(d, [blas, blas], [I, shifted]; refittable = true)

    @test tlas isa Mantle.HWTLAS          # Mantle's abstract type, not a Metal one
    @test tlas.count == 2
    @test tlas.handle.size > 0
    @test tlas.refittable

    # A refit rewrites the instance buffer and re-encodes. Checked by reading the
    # translation back out of the descriptor Metal will consume.
    moved = Mantle.Mat3x4f(1, 0, 0, 5,  0, 1, 0, 0,  0, 0, 1, 3)
    Mantle.refit_tlas!(d, tlas, [I, moved])
    ptr = convert(Ptr{MTL.MTLAccelerationStructureInstanceDescriptor},
                  MTL.contents(tlas.instances))
    second = unsafe_load(ptr, 2)
    @test reinterpret(Float32, [second.transformationMatrix])[10] == 5.0f0
    # The instance still names the same structure — a refit moves geometry, it
    # does not re-point instances.
    @test second.accelerationStructureIndex == UInt32(1)

    # Mismatched counts are rejected rather than silently truncating.
    @test_throws ArgumentError Mantle.refit_tlas!(d, tlas, [I])
end

@testset "Metal: a refit needs permission, and zero scratch is not refusal" begin
    d = Mantle.Device(Mantle.MetalAPI())
    verts = Float32[0, 0, 0,  1, 0, 0,  0, 1, 0]
    vbuf = MTL.MTLBuffer(d.dev, sizeof(verts); storage = Metal.SharedStorage)
    unsafe_copyto!(convert(Ptr{Float32}, MTL.contents(vbuf)), pointer(verts), length(verts))
    blas = Mantle.build_blas(d, vbuf, 1)

    plain = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()])
    @test !plain.refittable
    @test_throws ArgumentError Mantle.refit_tlas!(d, plain, [Mantle.identity_transform()])

    # The trap this pins: Metal reports `refitScratchBufferSize == 0` for a
    # structure this small even WITH `MTLAccelerationStructureUsageRefit`. Gating
    # the refit on "scratch > 0" therefore refuses a perfectly legal refit, which
    # is what the first version of this backend did.
    ok = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()]; refittable = true)
    @test ok.refittable
    Mantle.refit_tlas!(d, ok, [Mantle.Mat3x4f(1,0,0,2, 0,1,0,0, 0,0,1,0)])
    @test true                        # reached without throwing
end
