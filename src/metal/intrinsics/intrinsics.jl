"""
Device functions Mantle's kernels and render stages call that Metal.jl does not
provide: plain functions over AIR intrinsics, owned by Mantle.
"""
module MetalIntrinsics

using Metal
using Metal: AS

include("stages.jl")     # render stage builtins: fragment coordinates, derivatives, discard, point size
include("mesh.jl")       # mesh stage outputs
include("raytracing.jl") # ray queries
include("indirect.jl")   # rewriting an indirect command from a kernel

end
