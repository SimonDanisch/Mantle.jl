"""
Calls into the Metal API that Metal.jl does not wrap, as Mantle's own functions and
types. Nothing here adds a method to a function or a constructor of Metal.jl, so a
Metal.jl release that wraps the same call does not collide with it.
"""
module MetalBindings

using Metal
using Metal: MTL
using Metal.ObjectiveC: @objc, @objcwrapper, @objcproperties, id, nil

include("commands.jl")   # unmanaged command buffers and encoders, indirect dispatch
include("render.jl")     # render pipelines, render passes and draws
include("texture.jl")    # textures in heaps, mipmaps, readback
include("layer.jl")      # CAMetalLayer and its drawables
include("raytracing.jl") # acceleration structures
include("indirect.jl")   # indirect command buffers
include("mtl4.jl")       # Metal 4 command buffers, encoders and argument tables
include("mpsgraphs.jl")  # MPSGraph operations

end
