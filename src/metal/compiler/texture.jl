# Sampling a bound 2D texture, on Metal.
#
# `Lava.sample_texture_2d(binding, u, v, component)` names a SLOT, not a handle,
# because that is the shape a descriptor set gives on Vulkan: the texture is a
# module-level resource there and a shader reaches it by number. AIR has no such
# thing: a texture is a PARAMETER of the entry function, and a shader body several
# calls deep cannot name one. So the body emits a placeholder standing for "the
# thing bound at binding N" (`intrinsics/stages.jl`), and `lower_texture_bindings!`
# replaces it with the entry's parameter. That works because `finish_ir!` runs AFTER
# GPUCompiler's optimisation pipeline, by which point the whole shader is inlined
# into the entry and the parameter is in scope at the call site.
#
# ── The pointee type, and why a texture argument used to kill the compiler ───
#
# AIR is a TYPED-pointer dialect and the AGX compiler recognises a texture argument
# by its pointee's struct NAME:
#
#   define <4 x float> @TextureCopy(<4 x float>, <2 x float>,
#                                 %struct._texture_2d_t addrspace(1)* nocapture readonly)
#
# LLVM 22 has only opaque pointers, so that type has to be RECONSTRUCTED on the way
# down. `llvm-downgrade --bitcode-version=14.0` infers pointee types from uses, and a
# parameter whose only use is a call to a declared intrinsic gives it nothing to go
# on, so it emits `{} addrspace(1)*`, its placeholder for "unknown". Handed that,
# AGX looks up a texture descriptor for a struct that is not one, gets null, and
# SEGFAULTS: the crash is inside `AGXCompilerCore`, and `MTLRenderPipelineState`
# reports it as `XPC_ERROR_CONNECTION_INTERRUPTED` after retries, naming nothing.
#
# `byref(<ty>)` is what tells the downgrader. It is an ABI attribute LLVM carries on
# a pointer parameter for exactly this purpose (to say what the pointer points AT),
# and the downgrader reads it where it cannot infer:
#
#   %struct._texture_2d_t addrspace(1)* nocapture readonly byref(%struct._texture_2d_t)
#
# The call to the intrinsic gets a bitcast back to `{}`, because the DECLARATION's
# parameters are still unknown-pointee; AGX does not care, since what it reads is the
# entry's own signature and the function type in the `air.fragment` node, and both
# of those now name the struct.
#
# WHERE THAT WAS MEASURED. Not on a Julia shader: on Apple's. `TextureCopy` from
# CoreDisplay's metallib, packed with Metal.jl's own writer, builds a render pipeline
# from its ORIGINAL bytes; parsed by LLVM 22 and written back through the downgrader
# UNTOUCHED, the same function crashes the compiler service. The two disassemblies
# differ in nothing but `%struct._texture_2d_t`/`%struct._sampler_t` becoming `{}`
# in the signature, the call, the declaration and the metadata. Add `byref` and the
# round-tripped module builds again. So the bug was never in the shader this file
# emits, and an attribute on the parameter is the whole of the fix.
#
# An `elementtype(<ty>)` attribute is NOT read by the downgrader (measured: the
# parameter stays `{}` and the attribute is carried through unread), and neither is
# a zero-offset `getelementptr` naming the struct, which is fragile besides, since
# InstCombine folds one away. An attribute survives every pass that follows.

"""What AIR calls a texture parameter's type."""
air_texture_type_name(::Type{Texture2DPtr{T}}) where {T} =
    "texture2d<" * air_stage_name(T) * ", sample>"

"""
    texture_binding_parameters(job) -> (kinds, argtypes)

Per Julia argument: `1` where it is a texture parameter, `2` where it is a sampler,
`0` otherwise; and the argument types beside it.

Read off the SIGNATURE rather than from the LLVM types, because `ptr addrspace(1)`
is also what every buffer is and the two are only told apart by what the caller
declared.
"""
function texture_binding_parameters(@nospecialize(job::CompilerJob))
    args = collect(job.source.specTypes.parameters[2:end])
    job.config.params.stage === :mesh || pop!(args)   # the output pointer
    kinds = Int[a <: Texture2DPtr ? 1 : a === SamplerPtr ? 2 : 0 for a in args]
    return (kinds, args)
end

"""
    leading_offset(nparams, nmarkers) -> Int

How many parameters sit in FRONT of the first Julia argument.

A kernel ABI may prepend one that no Julia argument corresponds to, and
`stage_builtins!` appends one per referenced builtin, so the pad has to be computed
against the whole marker list, which covers the appended ones, and not against the
signature alone. Measuring from the end instead moves every binding one slot along
the moment a shader also reads `frag_coord`.
"""
function leading_offset(nparams::Int, nmarkers::Int)
    offset = nparams - nmarkers
    offset >= 0 ||
        error("$nparams parameters for $nmarkers markers: the marker list cannot be " *
              "longer than the signature it describes")
    return offset
end

"""
    lower_texture_bindings!(job, mod, entry, nmarkers)

Replace every "the texture/sampler at binding N" placeholder with the entry
parameter that holds it.

Parameters are matched to bindings BY ORDER: the n-th texture parameter of the
signature is binding `n - 1`, and likewise for samplers. That is the order the host
binds in, so a caller that declares its textures in the order it binds them needs
to know nothing else.
"""
function lower_texture_bindings!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                                 entry::LLVM.Function, nmarkers::Int)
    kinds, _ = texture_binding_parameters(job)
    any(!=(0), kinds) || return nothing
    params = collect(entry.parameters)
    offset = leading_offset(length(params), nmarkers)
    textures = LLVM.Value[params[i + offset] for i in eachindex(kinds) if kinds[i] == 1]
    samplers = LLVM.Value[params[i + offset] for i in eachindex(kinds) if kinds[i] == 2]
    name_texture_pointees!(mod, entry, kinds, offset)
    drop_alias_scopes!(mod)

    for (name, bound, what) in ((TEXTURE_PLACEHOLDER, textures, "texture"),
                                (SAMPLER_PLACEHOLDER, samplers, "sampler"))
        f = get(mod.functions, name, nothing)
        f === nothing && continue
        for use in collect(f.uses)
            call = use.user
            call isa LLVM.CallInst ||
                error("$name is referenced by something other than a call: $call")
            call.parent.parent === entry || error("""
                a $what binding is named from `$(call.parent.parent.name)`,
                which is not the stage entry. The replacement is an entry PARAMETER, so
                the call has to have been inlined into the entry; every shader is, by
                the time `finish_ir!` runs. A function this did not reach is one the
                optimiser kept out of line, and sampling from it needs the parameter
                threaded through instead.""")
            b = call.operands[1]
            b isa LLVM.ConstantInt || error("""
                the $what binding of a `sample_texture_2d` call is not a constant. It
                names an encoder slot, and the stage's parameter list is fixed at
                compile time, so the slot has to be known then. Compute the
                coordinates, not the binding.""")
            i = convert(Int, b)
            0 <= i < length(bound) || error("""
                this stage samples the $what at binding $i and was compiled for \
                $(length(bound)). A stage declares which bound textures it reads \
                (see `Mantle.FragmentShader`'s `textures`), and the list has to cover \
                every binding the body names.""")
            replace_uses!(call, bound[i + 1])
            erase!(call)
        end
        @assert isempty(f.uses) "$name still has uses after replacement"
        erase!(f)
    end
    return nothing
end

"""
    texture_argument_metadata!(arg_infos, kinds, argtypes, nmarkers)

Replace the `air.buffer` description of each texture and sampler parameter with its
own tag.

GPUCompiler described them as buffers, which is what a `ptr addrspace(1)` argument
is to the kernel ABI. Left that way, Metal binds a buffer where the stage expects a
texture.

The LEADING operand of each node is the parameter's POSITION, counted from zero, and
it has to name the parameter that actually holds the texture: the same alignment
`lower_texture_bindings!` uses to find it, from the same marker count.
"""
function texture_argument_metadata!(arg_infos::Vector, kinds::Vector{Int},
                                    @nospecialize(argtypes::Vector), nmarkers::Int)
    offset = leading_offset(length(arg_infos), nmarkers)
    tex = 0
    samp = 0
    for (i, k) in enumerate(kinds)
        k == 0 && continue
        pos = i + offset
        md = Metadata[Metadata(ConstantInt(Int32(pos - 1)))]
        if k == 1
            push!(md, MDString("air.texture"))
            loc = tex; tex += 1
            typename = air_texture_type_name(argtypes[i])
            argname = "texture$loc"
        else
            push!(md, MDString("air.sampler"))
            loc = samp; samp += 1
            typename = "sampler"
            argname = "sampler$loc"
        end
        append!(md, Metadata[MDString("air.location_index"),
                             Metadata(ConstantInt(Int32(loc))),
                             Metadata(ConstantInt(Int32(1)))])
        # `air.sample` says the texture is READ through a sampler rather than with
        # `read()`; a sampler entry carries no access tag at all.
        k == 1 && push!(md, MDString("air.sample"))
        append!(md, Metadata[MDString("air.arg_type_name"), MDString(typename),
                             MDString("air.arg_name"), MDString(argname)])
        arg_infos[pos] = MDNode(md)
    end
    return arg_infos
end

"""
    air_named_struct(mod, name) -> LLVM.StructType

The named, bodyless struct `name`, reused if the context already has one.

Reused and not recreated because a second `StructType("struct._texture_2d_t")` in
the same context is a DIFFERENT type named `struct._texture_2d_t.0`, and a stage
with two textures would then declare two unrelated pointee types.
"""
function air_named_struct(mod::LLVM.Module, name::String)
    T = get(mod.context.types, name, nothing)
    return T === nothing ? LLVM.StructType(name) : T
end

"""
    name_texture_pointees!(mod, entry, kinds, offset)

Say what each texture and sampler parameter POINTS AT, so the AIR downgrader can
reconstruct the typed pointer the driver reads.

A `byref(%struct._texture_2d_t)` attribute, for the reasons the header measures: it
is the one channel the downgrader reads, and unlike a zero-offset `getelementptr` it
survives the InstCombine that `stage_cleanup!` and `lower_air!` each end in.
"""
function name_texture_pointees!(mod::LLVM.Module, entry::LLVM.Function,
                                kinds::Vector{Int}, offset::Int)
    tex = samp = nothing
    for (i, k) in enumerate(kinds)
        k == 0 && continue
        if k == 1
            tex === nothing && (tex = air_named_struct(mod, "struct._texture_2d_t"))
            T = tex
        else
            samp === nothing && (samp = air_named_struct(mod, "struct._sampler_t"))
            T = samp
        end
        push!(entry.parameter_attributes[i + offset], TypeAttribute(:byref, T))
    end
    strip_debuginfo_for_byref!(mod)
    return nothing
end

"""
    drop_alias_scopes!(mod)

Remove every `!alias.scope` and `!noalias` attachment from a module that binds a
texture.

The macOS 27.0.1 compiler for the M5 (Apple10) crashes on a stage that samples a
texture and also loads through a pointer carrying Julia's scopes (`jnoalias_const`,
`jnoalias_stack`), and `MTLRenderPipelineState` reports it as
`XPC_ERROR_CONNECTION_INTERRUPTED`, like the pointee crash above. Either half alone
builds. Such loads read a stage argument passed by reference: GPUCompiler 2.14
passes one that already lives in the device address space through unchanged, so
Julia's own loads of it reach AIR with their metadata. 2.13 copied it into a stack
slot first, and the copy's plain load was all that reached the driver. Measured on
RayMakie's image fragment and on the by-reference texture test in
`test/metal/test_graphics_metal.jl` (2026-10-16): both build again once the two
attachments are gone.

Julia's optimizer has finished by the time this runs, so it loses nothing. Apple's
back end could still have used the scopes to reorder memory operations; that cost is
not measured. A stage that binds no texture keeps them.
"""
function drop_alias_scopes!(mod::LLVM.Module)
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        md = inst.metadata
        delete!(md, LLVM.MD_alias_scope)
        delete!(md, LLVM.MD_noalias)
    end
    return nothing
end

"""
    strip_debuginfo_for_byref!(mod)

Drop the module's debug info, which is what lets a `byref` on an OPAQUE struct reach
the downgrader.

`byref` is an ABI attribute, and LLVM's verifier rejects one whose type is unsized,
which every AIR handle type is, since `%struct._texture_2d_t` has no body and a
bodied struct in its place puts the AGX crash straight back (measured: `type {}` and
`type { i8 }` both segfault where `type opaque` builds). The verifier runs from
`UpgradeDebugInfo`, and `UpgradeDebugInfo` only verifies a module that declares the
CURRENT debug metadata version, which is why Apple's own shaders, carrying none, go
through untouched. Without the flag it strips the debug info instead, so this does
by hand what the downgrader would otherwise do to the same module a moment later.

The cost is line information for a stage that samples, and it is small: stages
compile at debug level 0, so the compile unit is `emissionKind: NoDebug` and nothing
was going to be emitted from it, and a graphics stage has no exception reporting to
attribute either. Nothing else in the module is touched, so a stage that samples no
texture keeps everything it had.
"""
function strip_debuginfo_for_byref!(mod::LLVM.Module)
    LLVM.strip_debuginfo!(mod)
    # …and the FLAG with it. `strip_debuginfo!` removes the compile unit and every
    # `!dbg` attachment but leaves `!{i32 2, "Debug Info Version", i32 3}` behind, and
    # that flag alone is what `UpgradeDebugInfo` reads to decide whether to verify.
    md = mod.metadata
    haskey(md, "llvm.module.flags") || return nothing
    flags = md["llvm.module.flags"]
    kept = LLVM.MDNode[]
    for node in collect(flags.operands)
        ops = collect(node.operands)
        isversion = length(ops) >= 2 && ops[2] isa LLVM.MDString &&
                    convert(String, ops[2]) == "Debug Info Version"
        isversion || push!(kept, node)
    end
    empty!(flags.operands)
    for node in kept
        push!(flags.operands, node)
    end
    return nothing
end
