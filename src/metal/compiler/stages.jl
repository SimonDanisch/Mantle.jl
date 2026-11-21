# Julia → AIR vertex and fragment programs.
#
# Metal.jl compiles Julia to compute kernels only: GPUCompiler's Metal target emits
# a `void` entry with buffer arguments and registers it under the `air.kernel` named
# metadata. This file rewrites such a kernel into a vertex or fragment program;
# `mesh.jl` does the same for mesh programs and `texture.jl` binds sampled textures.
# Metal.jl's metallib writer reads the program type off the named metadata the
# rewrite registers the entry under (`air.vertex`, `air.fragment`, `air.mesh`).
#
# ── The specification, read out of a shipping metallib ───────────────────────
#
# Not guessed. `/System/Library/Frameworks/PencilKit.framework/.../default.metallib`
# holds 25 functions, 7 of them `air.vertex` and 11 `air.fragment`; the shapes below
# are what Metal.jl's `MetalLib` reader plus LLVM.jl found in
# `sixChannelBlend_vertex` / `sixChannelBlend_fragment`. Reproduce with
# `read(io, Metal.MetalLib)` and `parse(LLVM.Module, fn.air_module)`.
#
# ENTRY SIGNATURE. Both stages RETURN a packed literal struct; this is the part that
# has no counterpart in the kernel path, where the entry is always `void`:
#
#   vertex   <{ <4 x float>, <2 x float>, <2 x float>, <2 x float> }>
#              (ptr addrspace(1), ptr addrspace(1), ptr addrspace(2), i32)
#   fragment <{ <4 x half>, <4 x half> }>
#              (<4 x float>, <2 x float>, …, ptr addrspace(2), …)
#
# The struct's fields correspond 1:1, in order, to the entries of the OUTPUT
# metadata list.
#
# NAMED METADATA. `air.vertex` / `air.fragment`, each node a triple with exactly the
# shape `air.kernel` already uses, `{ptr @entry, outputs, inputs}`, except that for a
# kernel the middle operand is an EMPTY node (GPUCompiler builds it as `stage_infos`,
# see its `metal.jl`), and here it carries the outputs.
#
# VERTEX OUTPUTS
#   {air.position,      air.arg_type_name, "float4", air.arg_name, "position"}
#   {air.vertex_output, "generated(<mangled>)",
#                       air.arg_type_name, "float2", air.arg_name, "texCoord"}
#
# FRAGMENT OUTPUTS
#   {air.render_target, i32 <index>, air.arg_type_name, "half4",
#                       air.arg_name, "color"}
#
# VERTEX INPUTS: buffers are IDENTICAL to what `add_argument_metadata!` already
# emits for kernels, so that machinery is reusable verbatim:
#   {i32 <n>, air.buffer, air.location_index, i32 <loc>, i32 1, air.read,
#    air.address_space, i32 <as>, air.arg_type_size, i32 <sz>,
#    air.arg_type_align_size, i32 <al>, air.arg_type_name, "<T>",
#    air.arg_name, "<name>"}
#   {i32 <n>, air.vertex_id, air.arg_type_name, "uint", air.arg_name, "vid"}
#
# FRAGMENT INPUTS
#   {i32 0, air.position, air.center, air.no_perspective, air.arg_type_name,
#    "float4", air.arg_name, "position"}
#   {i32 <n>, air.fragment_input, "generated(<mangled>)", air.center,
#    air.perspective, air.arg_type_name, "float2", air.arg_name, "texCoord"}
#   textures: {i32 <n>, air.texture, air.location_index, i32 <loc>, i32 1,
#              air.sample, air.arg_type_name, "texture2d<half, sample>", …}
#
# ── The one non-obvious thing: how the stages are wired together ─────────────
#
# A varying is matched between stages by the STRING in the second slot, and by
# nothing else: not by position, not by name, not by type. It reads
# `generated(16snapshotTexCoordDv2_f)`: an Itanium-style mangling of
# `<length><name><type>`, so `texCoord` of type `float2` becomes
# `generated(8texCoordDv2_f)`. Vertex output and fragment input must produce
# byte-identical strings or the two stages simply do not connect, and nothing
# reports it as an error.
#
# `mangle_varying` below is that rule, and `test/metal/test_graphics_metal.jl` pins
# it against the four strings observed in the reference metallib.

"""
    mangle_varying(name::AbstractString, T::Type) -> String

The `generated(...)` tag that links a vertex output to a fragment input.

Itanium-ish: the name with a length prefix, then the type encoding. This is the
ONLY thing that pairs the two stages, so vertex and fragment must derive it from
the same `(name, type)` pair or the varying silently carries garbage.
"""
function mangle_varying(name::AbstractString, T::Type)
    return "generated($(length(name))$(name)$(air_type_mangling(T)))"
end

"""
    air_type_mangling(T) -> String

Itanium mangling for the types a varying can have.

`f`/`h`/`i`/`j` are float/half/int/uint; `Dv<N>_` is the vector prefix, so a
`float2` is `Dv2_f`, which is what the reference metallib carries for its `float2`
varyings. Scalars have no prefix at all.
"""
air_type_mangling(::Type{Float32}) = "f"
air_type_mangling(::Type{Float16}) = "Dh"
air_type_mangling(::Type{Int32})   = "i"
air_type_mangling(::Type{UInt32})  = "j"
# `h` is Itanium's `unsigned char`, which is what a `device uchar*` points at.
# (`Dh` above is `half`; the two are distinct manglings that both start with an h.)
air_type_mangling(::Type{UInt8})   = "h"
air_type_mangling(::Type{Int8})    = "a"
air_type_mangling(::Type{NTuple{N,T}}) where {N,T} = "Dv$(N)_" * air_type_mangling(T)

"""
    air_wrapped_vector(T) -> Type or nothing

The `NTuple` inside a one-field vector wrapper, or `nothing` if `T` is not one.

`NTuple` is how a Julia shader spells a vector, but a caller writing a real shader
writes `Vec4f`, which is an `SVector{4,Float32}`, which is a struct whose single
field is an `NTuple{4,Float32}`. So the wrapper is UNWRAPPED rather than tabulated:
the same test serves `Point3f`, `RGBA` and whatever else a caller reaches for.

One field, and that field a homogeneous tuple, is the whole test. Anything else is
still an error, so a struct that merely happens to be small does not silently
become a vector.
"""
function air_wrapped_vector(@nospecialize(T::Type))
    (isconcretetype(T) && fieldcount(T) == 1) || return nothing
    F = fieldtype(T, 1)
    return F <: Tuple && isconcretetype(F) && F === NTuple{fieldcount(F),eltype(F)} ?
           F : nothing
end

function air_type_mangling(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing &&
        error("no AIR type mangling for $T: a varying must be a 32/16-bit scalar, " *
              "a vector of one, or a one-field wrapper around such a vector; " *
              "extend `air_type_mangling` if Metal grows another.")
    return air_type_mangling(F)
end

"""
    air_stage_name(T) -> String

The `air.arg_type_name` string for a stage input or output, e.g. `"float4"`.

Separate from the mangling because the two use different spellings for the same
type: the linkage tag says `Dv2_f`, the human-readable metadata says `float2`, and
the reference metallib carries both for every varying.
"""
air_stage_name(::Type{Float32}) = "float"
air_stage_name(::Type{Float16}) = "half"
air_stage_name(::Type{Int32})   = "int"
air_stage_name(::Type{UInt32})  = "uint"
air_stage_name(::Type{UInt8})   = "uchar"
air_stage_name(::Type{Int8})    = "char"
air_stage_name(::Type{NTuple{N,T}}) where {N,T} = air_stage_name(T) * string(N)
function air_stage_name(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing && error("no AIR type name for $T; see `air_type_mangling`")
    return air_stage_name(F)
end

# ── Rewriting a compiled kernel into a graphics stage ────────────────────────
#
# A stage is compiled as a KERNEL first, and only then reshaped. That is not a
# shortcut, it is the only order that works: every Metal-specific transform in
# GPUCompiler's `finish_ir!` (the parameter and global address-space rewrites,
# `split_aggregate_loads!`, `add_argument_metadata!`) is gated on
# `job.config.kernel`, and a vertex program needs all of them exactly as much as a
# compute one does. `MetalStageJob`'s `finish_ir!` invokes GPUCompiler's first, so
# this runs after that work is done and can reshape the result.
#
# The one thing a kernel cannot be is non-void, so the Julia side writes its stage
# output through a trailing pointer argument, kernel-style, and `stage_return!`
# turns that argument into the return value.
#
# It does NOT pattern-match the store. Guessing which instruction produces the
# output would break the moment the body writes it field by field, or in a branch,
# or twice. Instead the trailing parameter becomes an `alloca` that the cloned body
# still stores into, and every `ret void` becomes a load from that slot followed by
# a real `ret`. SROA in the clean-up promotes the slot away, so the emitted AIR
# looks as if the value had been returned all along.
#
# The clone idiom (conversion block, `value_map`, `clone_into!`, branch to the old
# entry, erase and rename) is GPUCompiler's own, from
# `add_parameter_address_spaces!`; deviating from it here would only find new ways
# to leave dangling metadata behind.

"""
    stage_output_type(job) -> Type

What the stage returns, read off the Julia signature's trailing pointer argument.

Derived rather than carried in `MetalStageParams`, because the two must agree and a
second source of truth is a second thing to get out of sync: the argument the body
writes through IS the declaration of what the stage outputs.
"""
function stage_output_type(@nospecialize(job::CompilerJob))
    sig = job.source.specTypes
    args = sig.parameters[2:end]          # [1] is the function itself
    isempty(args) &&
        error("a graphics stage needs a trailing output pointer argument; the signature has none")
    P = last(args)
    (P <: Core.LLVMPtr || P <: Ptr) && return P.parameters[1]
    error("the last argument of a graphics stage must be a pointer to its output " *
          "struct, got $P")
end

"""Whether the stage in `mod` calls `set_point_size!`."""
writes_point_size(mod::LLVM.Module) =
    haskey(mod.globals, POINT_SIZE_GLOBAL) && !isempty(mod.globals[POINT_SIZE_GLOBAL].uses)

"""
    stage_return!(job, mod, f; pointsize = false) -> LLVM.Function

Turn `f(args..., out::Ptr{T})` into `f(args...) -> T`.

With `pointsize`, the returned struct has one more field after `T`'s: a `float`
holding what `set_point_size!` stored, `[[point_size]]` once `retag_stage!` tags it.
The global it stored into becomes a stack slot, initialised to 1, and every exit
returns the slot's value in that field. Last rather than second, where SpriteKit's
`PointSprite_VertexFunc` puts it: the output list is matched to the fields by
position, so either is the same declaration.

Returns the new entry. `f` is erased and the new function takes its name, so
callers must re-look-up the entry afterwards.
"""
function stage_return!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function;
                       pointsize::Bool = false)
    ft = f.function_type
    params = collect(ft.parameters)
    isempty(params) &&
        error("a graphics stage needs a trailing output pointer argument; `$(f.name)` has none")
    outparam = last(params)
    outparam isa LLVM.PointerType ||
        error("the last argument of a graphics stage must be the output pointer, got $outparam")

    # What the stage returns, in AIR's spelling. NOT `convert(LLVMType, T)`: an
    # `NTuple{4,Float32}` lowers to `[4 x float]`, and AIR wants `<4 x float>`. The
    # reference metallib's vertex entry returns `<{ <4 x float>, <2 x float>, … }>`,
    # a PACKED struct of VECTORS, and a struct of arrays is a different type that
    # the Metal loader rejects.
    T_jl  = stage_output_type(job)
    T_out = air_output_struct(T_jl; pointsize)

    new_ft = LLVM.FunctionType(T_out, params[1:(end - 1)])
    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    for (arg, new_arg) in zip(f.parameters, new_f.parameters)
        new_arg.name = arg.name
    end

    slot = nothing
    psslot = nothing
    @dispose builder = IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, LLVM.at_end(entry))
        if pointsize
            psslot = alloca!(builder, LLVM.FloatType(), "point_size")
            store!(builder, LLVM.ConstantFP(1f0), psslot)
        end

        # The body keeps writing to a pointer; it just points at a stack slot now.
        # Cast into the address space the old parameter had, so nothing in the
        # cloned body sees a different pointer type than it was compiled against.
        # The slot keeps JULIA's layout, because that is what the cloned body stores
        # into; the conversion to AIR's vector form happens at the `ret`.
        slot = alloca!(builder, convert(LLVMType, T_jl), "stage_out")
        as = outparam.addrspace
        outptr = as == 0 ? slot : addrspacecast!(builder, slot, outparam)

        new_args = LLVM.Value[new_f.parameters[i] for i in 1:(length(params) - 1)]
        for i in 1:(length(params) - 1), attr in collect(f.parameter_attributes[i])
            push!(new_f.parameter_attributes[i], attr)
        end
        push!(new_args, outptr)

        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(f.parameters)
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes = LLVM.CloneFunctionChangeType.GlobalChanges)

        br!(builder, new_f.blocks[2])
    end

    # Every exit now returns the slot's contents.
    for bb in new_f.blocks, inst in collect(bb.instructions)
        inst isa LLVM.RetInst || continue
        isempty(inst.operands) || continue      # already returns something
        @dispose builder = IRBuilder() begin
            position!(builder, LLVM.before(inst))
            jl = load!(builder, convert(LLVMType, T_jl), slot)
            out = T_out isa LLVM.StructType ?
                  to_air_output(builder, jl, T_jl, T_out) :
                  to_air_field(builder, extract_value!(builder, jl, 0), fieldtype(T_jl, 1))
            if pointsize
                ps = load!(builder, LLVM.FloatType(), psslot)
                out = insert_value!(builder, out, ps, fieldcount(T_jl))
            end
            ret!(builder, out)
        end
        erase!(inst)
    end

    fn = f.name
    remove_dead_constant_users!(f)
    @assert isempty(f.uses) "graphics entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    new_f.name = fn
    if pointsize
        # Only the clone is left using the global now. Stages are inlined into their
        # entry, so a store anywhere else is a body this pass never saw.
        gv = mod.globals[POINT_SIZE_GLOBAL]
        for use in collect(gv.uses)
            u = use.user
            (u isa LLVM.StoreInst && LLVM.parent(LLVM.parent(u)) == new_f) ||
                error("@$POINT_SIZE_GLOBAL is used by a $(typeof(u)) outside the " *
                      "entry; only a store inlined into it is expected")
            LLVM.operands(u)[2] = psslot
        end
        erase!(gv)
    end
    return new_f
end

# ── The stage metadata ───────────────────────────────────────────────────────
#
# GPUCompiler already produced a well-formed `air.kernel` node for this entry:
# `{ptr @entry, <empty stage_infos>, <arg_infos>}`. A stage node has exactly the
# same three-operand shape, and its third operand, the argument metadata, is
# byte-for-byte what a kernel's is, because a buffer argument is described the same
# way whichever stage reads it (see the specification at the top of this file, and
# GPUCompiler's `add_argument_metadata!`).
#
# So this does not rebuild the node. It takes GPUCompiler's, keeps the argument
# list, drops the entry for the output pointer that `stage_return!` removed, fills
# in the outputs that a kernel leaves empty, and re-registers the whole thing under
# `air.vertex`, `air.fragment` or `air.mesh`. Rebuilding the argument metadata from
# scratch would mean re-deriving sizes, alignments and address spaces that
# GPUCompiler has already computed correctly.

"""Named metadata a stage is registered under."""
stage_metadata_key(stage::Symbol) =
    stage === :vertex ? "air.vertex" :
    stage === :fragment ? "air.fragment" :
    stage === :mesh ? "air.mesh" :
    error("no AIR metadata key for stage :$stage")

"""
    stage_outputs(stage, T) -> Vector{Metadata}

The output list: one node per field of the returned struct `T`.

For a vertex the first field is the clip position (`air.position`) and the rest are
varyings tagged with the `generated(...)` linkage string; for a fragment every
field is a render target. Field ORDER is the contract: the metadata list and the
struct's fields are matched positionally, not by name.
"""
function stage_outputs(stage::Symbol, @nospecialize(T::Type); pointsize::Bool = false)
    # A mesh stage returns nothing at all: everything it produces goes through the
    # object it was handed, so its `air.mesh` node carries an EMPTY output list.
    # That is the structural difference from the two stages below, and it is what
    # the shipping metallib shows.
    stage === :mesh && return Metadata[]
    names = fieldnames(T)
    isempty(names) && error("a graphics stage must return at least one value")
    out = Metadata[]
    for (i, name) in enumerate(names)
        F = fieldtype(T, i)
        md = Metadata[]
        if stage === :vertex && i == 1
            push!(md, MDString("air.position"))
        elseif stage === :vertex
            push!(md, MDString("air.vertex_output"))
            push!(md, MDString(mangle_varying(String(name), F)))
        else
            # TWO integers, not one: `{air.render_target, i32 <index>, i32 0, …}`.
            # Read off a literal-index fragment in the wild: `path_exterior_fragment`
            # carries `!{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", …}`.
            # PencilKit's has a function-constant pointer where the first integer
            # is, which is what hid the second one.
            #
            # Emitting only the index does not fail to build: the metallib loads and
            # reports `MTLFunctionTypeFragment`, and then creating a render pipeline
            # from it dies with "Internal compiler error (AGXMetalG17G, code 3)" and
            # nothing that names the cause.
            push!(md, MDString("air.render_target"))
            push!(md, Metadata(ConstantInt(Int32(i - 1))))
            push!(md, Metadata(ConstantInt(Int32(0))))
        end
        push!(md, MDString("air.arg_type_name"))
        push!(md, MDString(air_stage_name(F)))
        push!(md, MDString("air.arg_name"))
        push!(md, MDString(String(name)))
        push!(out, MDNode(md))
    end
    # The field `stage_return!` appended, spelled as SpriteKit's
    # `PointSprite_VertexFunc` spells it.
    pointsize && push!(out, MDNode(Metadata[MDString("air.point_size"),
                                            MDString("air.arg_type_name"),
                                            MDString("float"),
                                            MDString("air.arg_name"),
                                            MDString("point_size")]))
    return out
end

"""
    shift_buffer_location(node, by) -> MDNode

`node` with its `air.location_index` moved by `by`, or `node` unchanged if it does
not carry one.

Only an `air.buffer`/`air.texture` entry has a location; a builtin like
`air.thread_index_in_threadgroup` is not bound by the host and has none. The
LEADING operand of the node is the parameter position and is deliberately left
alone: a parameter does not move because a binding does.
"""
function shift_buffer_location(node::LLVM.MDNode, by::Integer)
    ops = collect(node.operands)
    i = findfirst(o -> o isa LLVM.MDString && convert(String, o) == "air.location_index", ops)
    i === nothing && return node
    i < length(ops) ||
        error("air.location_index is the last operand of an argument node; " *
              "it must be followed by the index itself")
    old = convert(Int, LLVM.Value(ops[i + 1]))
    ops[i + 1] = Metadata(ConstantInt(Int32(old + by)))
    return MDNode(ops)
end

"""
    retag_stage!(job, mod, entry, stage, T_out, markers; pointsize)

Move `entry` from `air.kernel` to the stage's own named metadata, with outputs.
"""
function retag_stage!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                      entry::LLVM.Function, stage::Symbol, @nospecialize(T_out::Type),
                      markers::Vector{Union{Nothing,Type}} = Union{Nothing,Type}[];
                      pointsize::Bool = false)
    md = mod.metadata
    haskey(md, "air.kernel") ||
        error("no air.kernel metadata to convert; did GPUCompiler's finish_ir! run?")

    # The node for THIS entry. A module can hold several after deferred codegen.
    kernel_nodes = collect(md["air.kernel"].operands)
    # The operand is the function wrapped as metadata, so the comparison has to be
    # made on that side, `Metadata(entry)`, rather than by unwrapping. By now
    # `replace_metadata_uses!` has already repointed the node at the rewritten
    # function, so this matches the NEW entry.
    want = Metadata(entry)
    idx = findfirst(kernel_nodes) do node
        ops = node.operands
        !isempty(ops) && ops[1] == want
    end
    idx === nothing &&
        error("entry $(entry.name) has no air.kernel node to convert")

    ops = collect(kernel_nodes[idx].operands)
    arg_infos = collect(ops[3].operands)

    # GPUCompiler described the ORIGINAL Julia signature. Two things happened to it
    # since: `stage_return!` dropped the trailing output pointer, and
    # `stage_builtins!` appended one parameter per referenced builtin. The metadata
    # has to follow both, or the entries stop describing the parameters they are
    # indexed against.
    isempty(arg_infos) && error("no argument metadata to adapt")
    # A mesh stage keeps every parameter it started with: nothing was dropped,
    # because there was no output pointer to drop.
    stage === :mesh || pop!(arg_infos)                # the output pointer
    nparams = length(entry.function_type.parameters)
    while length(arg_infos) < nparams
        # A placeholder per appended builtin; `stage_input_metadata!` replaces it
        # with the real tag below, since every one of them is a marked input.
        push!(arg_infos, MDNode(Metadata[]))
    end
    length(arg_infos) == nparams ||
        error("argument metadata does not match the rewritten signature: " *
              "$(length(arg_infos)) entries for $nparams parameters")

    isempty(markers) ||
        stage_input_metadata!(arg_infos, stage_align_markers(markers, length(arg_infos)))

    # A texture and a sampler are both `ptr addrspace(1)`/`ptr addrspace(2)` to the
    # kernel ABI, so GPUCompiler described them as buffers. Left that way Metal binds
    # a buffer where the stage expects a texture, and the pipeline fails to build
    # with an internal compiler error naming nothing.
    let (kinds, argtypes) = texture_binding_parameters(job)
        any(!=(0), kinds) &&
            texture_argument_metadata!(arg_infos, kinds, argtypes, length(markers))
    end

    # The object a mesh stage writes through. It is the FIRST parameter, and
    # GPUCompiler described it as a buffer, which is what the kernel ABI made it;
    # AIR needs `air.mesh` plus the whole type of the object, and that type is what
    # bounds the stage.
    #
    # Position 1 and not an offset from the end. Computing it from the end
    # overwrote whichever builtin happened to land there: with the compute builtins
    # threaded in, that was `air.thread_index_in_threadgroup`, so every thread in
    # the group read index 0, wrote to the same slots, and the stage behaved as if
    # it ran once. The object stayed described as a buffer at location 0 on top of
    # that, which is what made a stage buffer bound at Metal slot 0 collide with it.
    # Neither is a rule of Apple's: MSL puts a mesh stage's own buffer at
    # `buffer(0)` and it works.
    if stage === :mesh
        arg_infos[1] = air_mesh_argument(mesh_object_type(job))
        # …and the buffers behind it move down one slot, because the object is not
        # a buffer and does not consume a binding. So a mesh stage's buffers are
        # numbered from zero exactly like a vertex stage's, and nothing above this
        # backend has to know a mesh stage is different.
        for i in 2:length(arg_infos)
            arg_infos[i] = shift_buffer_location(arg_infos[i], -1)
        end
    end

    node = MDNode(Metadata[Metadata(entry),
                           MDNode(stage_outputs(stage, T_out; pointsize)),
                           MDNode(arg_infos)])
    push!(get!(md, stage_metadata_key(stage)).operands, node)

    # …and it must stop being a kernel. Leaving the old node behind would leave
    # `air.kernel` pointing at a function that now returns a struct, which is not a
    # kernel by any reading, and the metallib writer would describe it as one.
    # Named metadata has no per-operand removal, so the list is emptied and the
    # OTHER entries (deferred codegen can leave several in one module) are pushed
    # back.
    kept = [n for (i, n) in enumerate(kernel_nodes) if i != idx]
    empty!(md["air.kernel"].operands)
    for n in kept
        push!(md["air.kernel"].operands, n)
    end
    return nothing
end

"""
    air_output_struct(T) -> LLVM.StructType

The AIR type a stage returns: a PACKED struct whose fields are LLVM vectors.

`convert(LLVMType, ::NTuple{4,Float32})` gives `[4 x float]`, an ARRAY. AIR spells
the same thing `<4 x float>`, and the reference metallib's entries return
`<{ <4 x float>, <2 x float>, … }>`. The two are distinct LLVM types and the loader
only accepts the second.
"""
function air_output_struct(@nospecialize(T::Type); pointsize::Bool = false)
    fields = LLVMType[]
    for i in 1:fieldcount(T)
        push!(fields, air_field_type(fieldtype(T, i)))
    end
    pointsize && push!(fields, LLVM.FloatType())
    # A single output is still WRAPPED here, though every shipped single-target
    # fragment returns the vector bare: `TextureCopy` returns `<4 x float>`, and the
    # packed struct appears only with several fields. The one-field struct is what
    # this backend has always emitted and what every test draws through, so it is
    # left alone: returning it bare was tried while chasing the texture crash in
    # `texture.jl` and changed nothing.
    return LLVM.StructType(fields; packed = true)
end

"""AIR's type for one stage output field: a vector for an `NTuple`, else itself."""
air_field_type(::Type{NTuple{N,T}}) where {N,T} =
    LLVM.VectorType(convert(LLVMType, T), N)
function air_field_type(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    return F === nothing ? convert(LLVMType, T) : air_field_type(F)
end

"""
    to_air_output(builder, val, T_jl, T_air) -> LLVM.Value

Repack a Julia-shaped output value into AIR's vector-of-fields form.

Field by field, and element by element within a field, because there is no bitcast
between `[4 x float]` and `<4 x float>`: LLVM treats an aggregate and a vector as
unrelated. The shuffle is free after `instcombine`; what matters is that the
emitted type is the one AIR names.
"""
function to_air_output(builder::IRBuilder, val::LLVM.Value,
                       @nospecialize(T_jl::Type), T_air::LLVM.StructType)
    out = LLVM.UndefValue(T_air)
    for i in 1:fieldcount(T_jl)
        F = fieldtype(T_jl, i)
        fld = extract_value!(builder, val, i - 1)
        out = insert_value!(builder, out, to_air_field(builder, fld, F), i - 1)
    end
    return out
end

"""One output field: an `NTuple` becomes a vector, anything else passes through."""
function to_air_field(builder::IRBuilder, val::LLVM.Value, ::Type{NTuple{N,T}}) where {N,T}
    vec = LLVM.UndefValue(LLVM.VectorType(convert(LLVMType, T), N))
    for j in 1:N
        elt = extract_value!(builder, val, j - 1)
        vec = insert_element!(builder, vec, elt, ConstantInt(Int32(j - 1)))
    end
    return vec
end
# A wrapper (`Vec4f` and friends) is one `extract_value!` away from the tuple the
# method above handles. LLVM sees `{ [4 x float] }`, so the field has to come out
# before the elements can.
function to_air_field(builder::IRBuilder, val::LLVM.Value, @nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing && return val
    return to_air_field(builder, extract_value!(builder, val, 0), F)
end

# ── Stage inputs ─────────────────────────────────────────────────────────────
#
# The markers (`VertexID`, `InstanceID`, `FragCoord`, `Varying`, defined with the
# intrinsics) say which parameters arrive as VALUES. `stage_inputs!` turns each such
# parameter into a bare scalar, spilling it to a stack slot so the cloned body's
# loads still typecheck: the same trick as the output, in reverse; SROA removes the
# slot again.

"""What `air.*` tag a stage-input marker carries, and how AIR names its type."""
stage_input_tag(::Type{VertexID})   = ("air.vertex_id",   "uint")
stage_input_tag(::Type{InstanceID}) = ("air.instance_id", "uint")
stage_input_tag(::Type{FragCoord})  = ("air.position",    "float4")
stage_input_tag(::Type{Varying{name,T}}) where {name,T} =
    ("air.fragment_input", air_stage_name(T))

"""The scalar an input marker is passed as."""
stage_input_llvmtype(::Type{VertexID})   = convert(LLVMType, UInt32)
stage_input_llvmtype(::Type{InstanceID}) = convert(LLVMType, UInt32)
stage_input_llvmtype(::Type{FragCoord})  = LLVM.VectorType(convert(LLVMType, Float32), 4)
stage_input_llvmtype(::Type{Varying{name,T}}) where {name,T} = air_field_type(T)

"""
    stage_input_types(job) -> Vector{Union{Nothing,Type}}

Per Julia argument: the marker type if it is a stage input, `nothing` otherwise.

The trailing output pointer is excluded: `stage_return!` has already removed it by
the time this matters.
"""
function stage_input_types(@nospecialize(job::CompilerJob))
    args = collect(job.source.specTypes.parameters[2:end])
    # A mesh stage has no trailing output pointer (it writes through the object that
    # is its FIRST argument), so there is nothing to drop here, and that argument is
    # retagged by `air_mesh_argument` rather than marked.
    job.config.params.stage === :mesh || pop!(args)
    inputs = Union{VertexID, InstanceID, FragCoord, Varying}
    return Union{Nothing,Type}[a <: inputs ? a : nothing for a in args]
end

"""
    stage_align_markers(markers, nparams) -> Vector{Union{Nothing,Type}}

Right-align the per-Julia-argument markers against the LLVM parameter list.

A kernel ABI may prepend an implicit parameter that no Julia argument corresponds
to (a kernel-state pointer; a stage job has none today), so parameter `i` is Julia
argument `i - offset`. Padding at the front rather than assuming a fixed offset
keeps this correct if the ABI grows or drops an implicit parameter.
"""
function stage_align_markers(markers::Vector{Union{Nothing,Type}}, nparams::Int)
    offset = nparams - length(markers)
    offset >= 0 ||
        error("$(length(markers)) markers for only $nparams parameters")
    return Union{Nothing,Type}[i <= offset ? nothing : markers[i - offset]
                               for i in 1:nparams]
end

"""
    stage_inputs!(job, mod, f, markers) -> LLVM.Function

Turn every parameter named by `markers` from a buffer pointer into a value.

`markers[i]` is the marker type for parameter `i`, or `nothing` to leave it alone.
The value is spilled to a stack slot and the slot handed to the cloned body, so
nothing inside has to know the calling convention changed.
"""
function stage_inputs!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                       f::LLVM.Function, markers::Vector{Union{Nothing,Type}})
    any(!isnothing, markers) || return f

    ft = f.function_type
    params = collect(ft.parameters)
    markers = stage_align_markers(markers, length(params))

    # A parameter that ALREADY has the scalar type is one `stage_builtins!`
    # appended: those arrive as values, not as buffer pointers, so there is nothing
    # to convert, and marking them again would try to read the `.addrspace` of an
    # `i32`. Only the marker-typed Julia arguments need the rewrite.
    needs = [markers[i] !== nothing && params[i] != stage_input_llvmtype(markers[i])
             for i in eachindex(params)]
    any(needs) || return f          # nothing left to convert
    new_types = LLVMType[needs[i] ? stage_input_llvmtype(markers[i]) : params[i]
                         for i in eachindex(params)]
    new_ft = LLVM.FunctionType(ft.return_type, new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    for (arg, new_arg) in zip(f.parameters, new_f.parameters)
        new_arg.name = arg.name
    end

    @dispose builder = IRBuilder() begin
        entry = BasicBlock(new_f, "stage_inputs")
        position!(builder, LLVM.at_end(entry))

        new_args = LLVM.Value[]
        for (i, param) in enumerate(params)
            arg = new_f.parameters[i]
            if !needs[i]
                push!(new_args, arg)
                for attr in collect(f.parameter_attributes[i])
                    push!(new_f.parameter_attributes[i], attr)
                end
            else
                # The body was compiled against a pointer to the marker struct, which
                # is a one-field wrapper around the scalar, so a slot holding the
                # scalar has the same layout, and a cast to the old parameter's
                # address space keeps every load in the body valid.
                slot = alloca!(builder, arg.value_type)
                store!(builder, arg, slot)
                as = param.addrspace
                push!(new_args, as == 0 ? slot : addrspacecast!(builder, slot, param))
            end
        end

        value_map = Dict{LLVM.Value, LLVM.Value}(
            p => new_args[i] for (i, p) in enumerate(f.parameters)
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes = LLVM.CloneFunctionChangeType.GlobalChanges)
        br!(builder, new_f.blocks[2])
    end

    fn = f.name
    remove_dead_constant_users!(f)
    @assert isempty(f.uses) "stage entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    new_f.name = fn
    return new_f
end

"""
    stage_input_metadata!(arg_infos, markers)

Replace the `air.buffer` description of each stage input with its own tag.

GPUCompiler described those arguments as buffers because that is what the kernel
ABI made them; after `stage_inputs!` they are values, and the metadata has to say
so or Metal binds a buffer to a register operand.
"""
function stage_input_metadata!(arg_infos::Vector, markers::Vector{Union{Nothing,Type}})
    for (i, m) in enumerate(markers)
        m === nothing && continue
        tag, typename = stage_input_tag(m)
        md = Metadata[Metadata(ConstantInt(Int32(i - 1))), MDString(tag)]
        argname = string(nameof(m))
        # A fragment's interpolated position also declares how it is sampled; the
        # reference metallib pairs `air.position` with `air.center` and
        # `air.no_perspective`.
        if m === FragCoord
            push!(md, MDString("air.center"))
            push!(md, MDString("air.no_perspective"))
        elseif m <: Varying
            # The linkage string first, then the sampling: the order the reference
            # metallib's fragment inputs carry:
            #   {i32 n, air.fragment_input, "generated(…)", air.center,
            #    air.perspective, air.arg_type_name, "float2", air.arg_name, …}
            name, T = m.parameters
            push!(md, MDString(mangle_varying(string(name), T)))
            push!(md, MDString("air.center"))
            push!(md, MDString("air.perspective"))
            argname = string(name)
        end
        append!(md, Metadata[MDString("air.arg_type_name"), MDString(typename),
                             MDString("air.arg_name"), MDString(argname)])
        arg_infos[i] = MDNode(md)
    end
    return arg_infos
end

"""
    stage_cleanup!(mod)

Promote the slots the rewrites introduced, and with them the address-space casts.

Both passes hand the cloned body a pointer where it expects one (an `alloca` for
the output, another for each stage input) and cast it into the address space the
parameter had. That cast is `thread` to `device`, which is not a legal thing to hold
on Metal: the store goes to a stack slot the device pointer only pretends to name.

It is meant to be temporary. SROA promotes the slot into registers and the cast
disappears with it, leaving what the reference shaders contain: a value built with
`insertvalue` and returned, no memory at all:

    %13 = insertvalue <{ <4 x float> }> undef, <4 x float> %12, 0
    ret <{ <4 x float> }> %13

Without this the vertex program still compiles, still links, and still builds a
render pipeline, and draws nothing, because the position was written to a stack
slot through a device pointer. Nothing reports it.

The function passes are GPUCompiler's own from `add_parameter_address_spaces!`,
which introduces the same shape and cleans up after itself the same way. The
always-inliner in front of them folds the exception signal that
`stage_drop_exception_signal!` emptied into its callers.
"""
function stage_cleanup!(mod::LLVM.Module)
    @dispose pb = LLVM.PassBuilder() begin
        LLVM.add!(pb, LLVM.AlwaysInlinerPass())
        LLVM.add!(pb, LLVM.FunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.SimplifyCFGPass())
            LLVM.add!(fpm, LLVM.SROAPass())
            LLVM.add!(fpm, LLVM.EarlyCSEPass())
            LLVM.add!(fpm, LLVM.InstCombinePass())
        end
        LLVM.run!(pb, mod)
    end
    return nothing
end

"""
    stage_drop_exception_signal!(mod) -> Bool

Give the exception-signalling runtime function an empty body.

A stage job does not use exception sites (`uses_exception_sites` is false), so
GPUCompiler ends every throw site it lowers with a call to `gpu_signal_exception`,
unconditionally: the debug level gates the REPORTING calls before it, not this one.
Metal.jl's runtime library has no `signal_exception` (its own kernels report throw
sites instead, through the kernel state), so the call names a declaration nothing
defines, and IR validation rejects the module as calling an unknown function. A
definition that wrote a mailbox would not help either: a graphics stage has no
kernel state to reach one through, and no host call reads one after a draw.

So the function gets an empty body and the `llvm.trap` GPUCompiler emits right
after the call stays, which is what actually stops an out-of-bounds lane (until
`lower_air!` turns it into a clean return). Returns whether the module signals.

Run BEFORE `stage_cleanup!`, so the now-empty function is inlined away rather than
left as a call in the shader's hot path.
"""
function stage_drop_exception_signal!(mod::LLVM.Module)
    f = get(mod.functions, "gpu_signal_exception", nothing)
    f === nothing && return false
    isdeclaration(f) || empty!(f)
    @dispose builder = IRBuilder() begin
        position!(builder, LLVM.at_end(BasicBlock(f, "entry")))
        ret!(builder)
    end
    # It does nothing now, and saying so lets the inliner and DCE treat it as the
    # no-op it is instead of a call that might write memory.
    push!(f.function_attributes, EnumAttribute(:alwaysinline))
    # …and PRIVATE, so it is not exported. A stage compiled on its own never
    # noticed: its metallib is linked with nothing. A function linked INTO another
    # pipeline that carries its own `gpu_signal_exception` would give two external
    # definitions of the same name, which the Metal linker refuses with "symbol
    # multiply defined" and no hint that an emptied helper is what collided.
    f.linkage = LLVM.Linkage.Private
    return true
end

# ── Implicit stage inputs ────────────────────────────────────────────────────
#
# `vertex_index()` is a zero-argument call in the shader and a PARAMETER in AIR.
# Something has to bridge that, because AIR has no ambient vertex id: it is an entry
# argument tagged `air.vertex_id` and nothing else.
#
# The shader-side half is `intrinsics/stages.jl`: each builtin loads an external
# global that never gets a definition. This pass is the other half. For every such
# global the module actually references, it appends a parameter of the matching
# type, replaces the loads with it, and reports the marker so `retag_stage!` tags
# the argument.
#
# Same shape as Lava's `addrspace(7)` globals, which its SPIR-V emitter turns into
# BuiltIn Input variables, so a shader written against `vertex_index()` and
# `frag_coord_x()` compiles unchanged on either backend.

"""
    stage_builtins!(job, mod, f) -> (LLVM.Function, Vector{Type})

Turn every referenced stage-builtin global into a trailing entry parameter.

Returns the rewritten entry and the marker types, in the order the parameters were
appended, for `stage_input_types` to concatenate.

Only globals that are USED are turned into parameters: an unreferenced builtin
would otherwise cost a register and, worse, an argument slot that shifts every
buffer index after it.
"""
function stage_builtins!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                         f::LLVM.Function)
    builtins = (VERTEX_ID_GLOBAL => VertexID, INSTANCE_ID_GLOBAL => InstanceID,
                FRAG_COORD_GLOBAL => FragCoord)
    used = Tuple{String,Type,LLVM.GlobalVariable}[]
    for (name, marker) in builtins
        haskey(mod.globals, name) || continue
        gv = mod.globals[name]
        isempty(gv.uses) && continue
        push!(used, (name, marker, gv))
    end
    isempty(used) && return (f, Type[])
    # Deterministic order, by name: the parameter order IS the argument-metadata
    # order, so it must not vary between runs.
    sort!(used; by = first)

    ft = f.function_type
    params = collect(ft.parameters)
    extra = LLVMType[stage_input_llvmtype(m) for (_, m, _) in used]
    new_ft = LLVM.FunctionType(ft.return_type, vcat(params, extra))
    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    for (arg, new_arg) in zip(f.parameters, new_f.parameters)
        new_arg.name = arg.name
    end
    for (i, (name, _, _)) in enumerate(used)
        new_f.parameters[length(params) + i].name = name
    end

    value_map = Dict{LLVM.Value, LLVM.Value}(
        p => new_f.parameters[i] for (i, p) in enumerate(f.parameters)
    )
    value_map[f] = new_f
    clone_into!(new_f, f; value_map,
                changes = LLVM.CloneFunctionChangeType.GlobalChanges)

    # Every load of the builtin becomes a use of the new parameter. The load is what
    # the `llvmcall` in the intrinsic emitted; there is nothing else the global can
    # appear in, since it has no definition to store into.
    for (i, (name, _, _)) in enumerate(used)
        gv = mod.globals[name]
        arg = new_f.parameters[length(params) + i]
        for use in collect(gv.uses)
            u = use.user
            u isa LLVM.LoadInst ||
                error("stage builtin @$name is used by $(typeof(u)); only a load is expected")
            replace_uses!(u, arg)
            erase!(u)
        end
    end

    fn = f.name
    remove_dead_constant_users!(f)
    @assert isempty(f.uses) "stage entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    new_f.name = fn

    # The globals are dead now; leaving them behind would make the module declare an
    # undefined symbol the loader has to resolve.
    for (name, _, _) in used
        gv = mod.globals[name]
        isempty(gv.uses) && erase!(gv)
    end

    return (new_f, Type[m for (_, m, _) in used])
end

"""
    stage_builtins_before_output!(job, mod, f) -> (LLVM.Function, Vector{Type})

`stage_builtins!`, with the output pointer kept in last place.

Builtins are appended at the end, and `stage_return!` takes the LAST parameter to be
the output pointer. Running them in the naive order would leave the output buried
and a builtin holding its place. So the output parameter is moved back to the end
after the append: cheaper and far clearer than teaching `stage_return!` to look for
it somewhere else. A mesh stage has no output pointer, so nothing moves.
"""
function stage_builtins_before_output!(@nospecialize(job::CompilerJob),
                                       mod::LLVM.Module, f::LLVM.Function)
    new_f, markers = stage_builtins!(job, mod, f)
    (isempty(markers) || job.config.params.stage === :mesh) && return (new_f, markers)
    return (rotate_output_last!(mod, new_f, length(markers)), markers)
end

"""
    rotate_output_last!(mod, f, ntrailing) -> LLVM.Function

Move the parameter sitting `ntrailing` from the end back to the end.

The output pointer was last until `stage_builtins!` appended to the signature.
"""
function rotate_output_last!(mod::LLVM.Module, f::LLVM.Function, ntrailing::Int)
    ft = f.function_type
    params = collect(ft.parameters)
    out_idx = length(params) - ntrailing
    order = vcat(1:(out_idx - 1), (out_idx + 1):length(params), out_idx)

    new_ft = LLVM.FunctionType(ft.return_type, LLVMType[params[i] for i in order])
    new_f = LLVM.Function(mod, "", new_ft)
    new_f.linkage = f.linkage
    for (new_pos, old_pos) in enumerate(order)
        new_f.parameters[new_pos].name = f.parameters[old_pos].name
    end

    value_map = Dict{LLVM.Value, LLVM.Value}(
        f.parameters[old_pos] => new_f.parameters[new_pos]
        for (new_pos, old_pos) in enumerate(order)
    )
    value_map[f] = new_f
    clone_into!(new_f, f; value_map,
                changes = LLVM.CloneFunctionChangeType.GlobalChanges)

    fn = f.name
    remove_dead_constant_users!(f)
    @assert isempty(f.uses) "stage entry $(fn) still has uses after reordering"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    new_f.name = fn
    return new_f
end

"""
    stage_passes!(job, mod, entry) -> LLVM.Function

Rewrite the kernel GPUCompiler's `finish_ir!` made of `job` into the stage it
names, and return the new entry.

A vertex or fragment program is a kernel up to this point and a stage from here
on: GPUCompiler's `finish_ir!` is what applied the address-space rewrites and
emitted the argument metadata, and both are needed either way.
"""
function stage_passes!(@nospecialize(job::CompilerJob), mod::LLVM.Module, entry::LLVM.Function)
    stage = job.config.params.stage
    T_out = stage === :mesh ? Nothing : stage_output_type(job)
    markers = stage_input_types(job)
    # Builtins first: `vertex_index()` and friends become TRAILING parameters, so
    # they have to be appended while the output pointer is still last; otherwise
    # they would land after it and `stage_return!` would drop the wrong one.
    entry, implicit = stage_builtins_before_output!(job, mod, entry)
    append!(markers, implicit)
    # Output next: it drops the trailing pointer, so the marker list (which already
    # excludes it) lines up with the parameters afterwards.
    #
    # A MESH stage has none. Its entry stays `void` and everything it produces goes
    # through the object it was handed, so there is no return value to build and
    # `retag_stage!` retags that object's argument metadata instead.
    #
    # Asked before it runs, because it consumes the global `set_point_size!` stored
    # into; `retag_stage!` then tags the field it appended.
    pointsize = stage === :vertex && writes_point_size(mod)
    stage === :mesh || (entry = stage_return!(job, mod, entry; pointsize))
    entry = stage_inputs!(job, mod, entry, markers)
    # Textures last of the signature rewrites: the placeholders a shader body emits
    # for "the texture at binding N" are replaced with the entry's own parameters,
    # so the entry has to be the final one. See `texture.jl` for why this cannot be
    # an argument the body takes.
    lower_texture_bindings!(job, mod, entry, length(markers))
    retag_stage!(job, mod, entry, stage, T_out, markers; pointsize)
    # A stage has no kernel state, so the throw sites GPUCompiler lowered cannot
    # signal through one. Before the cleanup, so the emptied function is inlined
    # away.
    stage_drop_exception_signal!(mod)
    # …and then get rid of the slots the passes introduced, before the
    # thread-to-device casts around them reach the driver.
    stage_cleanup!(mod)
    return mod.functions[entry.name]
end
