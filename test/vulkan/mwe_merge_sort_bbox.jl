# MWE: AcceleratedKernels.merge_sort! on LavaArray{BBox{Float32}} crashes
# with SPIR-V validation error:
#   OpPtrAccessChain result type (OpTypeStruct) does not match the type
#   that results from indexing into the base <id> (OpTypeInt).
#
# From ImplicitBVH.BVH construction → AK.sort! → AK.merge_sort!
#
# Run: julia --project=. dev/Lava/test/mwe_merge_sort_bbox.jl

using Lava, Mantle
using ImplicitBVH: BBox
import AcceleratedKernels as AK
import KernelAbstractions as KA

backend = Mantle.LavaBackend()

# Create random bounding boxes
N = 1024
bboxes_cpu = [BBox(rand(Float32, 3)..., rand(Float32, 3)...) for _ in 1:N]
bboxes_gpu = KA.allocate(backend, BBox{Float32}, N)
copyto!(bboxes_gpu, bboxes_cpu)

println("Sorting $(N) BBox{Float32} on $(typeof(backend))...")
AK.sort!(bboxes_gpu)
println("OK!")
