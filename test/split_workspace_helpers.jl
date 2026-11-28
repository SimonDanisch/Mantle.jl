using Mantle, KernelAbstractions

# Supply a workspace factory only when replaying the frozen legacy policy.
@noinline function with_split_programs(f, backend; workspace=nothing)
    M,K = 256,1024
    A = Mantle.devicearray(backend,ones(Float32,M,K))
    A2 = Mantle.devicearray(backend,fill(2f0,M,K))
    B = Mantle.devicearray(backend,ones(Float32,K))
    C = Mantle.devicearray(backend,zeros(Float32,M))
    C2 = Mantle.devicearray(backend,zeros(Float32,M))
    S = Mantle.gemv_split(C,A,B,M,K)
    kw(C) = workspace === nothing ? (;) : (;partials=workspace(backend,C,M,1,S))
    first = Mantle.scalar_gemm_launches(C,A,B,M,1,K,1f0,0f0;kw(C)...)
    second = Mantle.scalar_gemm_launches(C2,A2,B,M,1,K,1f0,0f0;kw(C2)...)
    @assert length(first)==length(second)==2
    state = (;outputs=(C,C2),expected=(fill(Float32(K),M),fill(Float32(2K),M)),
             workspaces=(first[1].args[1],second[1].args[1]),
             launches=(first[1],second[1],first[2],second[2]))
    try
        return f(state)
    finally
        KernelAbstractions.synchronize(backend)
        foreach(Mantle.unsafe_free!,(A,A2,B,C,C2))
    end
end
