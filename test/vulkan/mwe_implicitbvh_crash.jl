# MWE: Lava SPIR-V emitter produces invalid OpPtrAccessChain when sorting
# composite structs with `by=` accessor in shared memory.
#
# Error: OpPtrAccessChain result type (OpTypeStruct) does not match the type
#        that results from indexing into the base <id> (OpTypeInt).
#
# From AcceleratedKernels.merge_sort! with `by=bv->bv.morton` on a struct
# array in shared memory. The emitter generates wrong pointer types when
# accessing fields of structs stored in Workgroup (shared) memory.
#
# Run: julia --project=. dev/Lava/test/mwe_implicitbvh_crash.jl

using Lava, Mantle
using ImplicitBVH: BoundingVolume, BBox
import AcceleratedKernels as AK
import KernelAbstractions as KA

backend = Mantle.defaultbackend()

BV = BoundingVolume{BBox{Float32}, Int32, UInt32}

bvs_cpu = [BoundingVolume(
    BBox((rand(Float32), rand(Float32), rand(Float32)),
         (rand(Float32)+1, rand(Float32)+1, rand(Float32)+1)),
    Int32(i),
    rand(UInt32)
) for i in 1:256]

bvs_gpu = KA.allocate(backend, BV, 256)
copyto!(bvs_gpu, bvs_cpu)

println("Sorting BoundingVolume by morton code...")
AK.sort!(bvs_gpu; by=bv->bv.morton)
println("OK!")
