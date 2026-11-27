# `norm` of a device array, for the backends' array types (`LinearAlgebra.norm` of a
# `LavaArray` or a WebGPU array calls it).

"""
    devicenorm(v, p = 2)

`LinearAlgebra.norm(v, p)` of a device array. Replaces GPUArrays' generic `norm`,
whose closures capture `Core.Box`es that do not compile for a GPU, and whose
rescaling divides by `maxabs`: drivers lower `x / m` to `x * (1/m)`, and for `m`
near `floatmax` that reciprocal is subnormal and flushed to zero (lavapipe, RADV),
so the norm comes out 0. Here a device with Float64 accumulates Float16 and
Float32 in it, where nothing overflows; otherwise the sum is rescaled by a
normal power of two based on the largest magnitude. `p` other
than 0, 1, 2 and ±Inf needs transcendental math that is inaccurate on some drivers
(RDNA3.5), so it runs on the host.
"""
function devicenorm(v::AbstractArray{T}, p::Real = 2) where {T}
    RT = typeof(float(LinearAlgebra.norm(zero(T))))
    isempty(v) && return zero(RT)
    # `Base.count`: the bare name is Mantle's resource element `count`.
    p == 0 && return convert(RT, Base.count(!iszero, v))
    p == Inf && return convert(RT, maximum(abs, v))
    p == -Inf && return convert(RT, minimum(abs, v))
    (p != 1 && p != 2) && return convert(RT, LinearAlgebra.norm(Array(v), p))
    float64 = supports_float64(Device(KernelAbstractions.get_backend(v)))
    if float64 && (RT === Float32 || RT === Float16)
        p == 1 && return RT(sum(x -> Float64(abs(x)), v; init = 0.0))
        return RT(sqrt(sum(x -> Float64(abs(x))^2, v; init = 0.0)))
    end
    # The accumulation type: Float64 where there is one, else at least Float32.
    return devicenorm(v, p, RT, float64 ? promote_type(Float64, RT) : promote_type(Float32, RT))
end

# A static parameter, so the kernels' closures capture no `DataType`.
function devicenorm(v::AbstractArray, p::Real, ::Type{RT}, ::Type{A}) where {RT, A}
    p == 1 && return convert(RT, sum(x -> A(abs(x)), v; init = zero(A)))
    maxabs = A(maximum(abs, v))
    (iszero(maxabs) || !isfinite(maxabs)) && return convert(RT, maxabs)
    # Unscaled when the sum of squares can neither overflow nor underflow.
    squarescale = maxabs^2
    isfinite(length(v) * squarescale) && squarescale >= floatmin(A) &&
        return convert(RT, sqrt(sum(x -> A(abs(x))^2, v; init = zero(A))))
    minexp = exponent(floatmin(A))
    shift = clamp(-exponent(maxabs), minexp, -minexp)
    scale = ldexp(one(A), shift)
    scaled = sqrt(sum(x -> (A(abs(x)) * scale)^2, v; init = zero(A)))
    return convert(RT, ldexp(scaled, -shift))
end
