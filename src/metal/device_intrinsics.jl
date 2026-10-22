# `KernelInterface`'s device half for Metal. Lane queries, `localmemory`, `shfl`,
# `sub_group_reduce_add` and barriers come from Metal.jl's own KI methods; an
# override here would shadow them. A capability Metal.jl does not wrap is added
# to Metal.jl, not worked around here.

"""
    StageOut{T}

Trailing output-pointer argument of every Metal shader stage wrapper here, which
Metal's `stage_return!` replaces with a stack slot it returns. Address space 0: a
slot `addrspacecast` to device space (`1`) is different memory on Apple GPUs, so
stores can miss it and a fragment silently returns zeros (`test_tuple_varyings.jl`).
"""
const StageOut{T} = Core.LLVMPtr{T,0}
