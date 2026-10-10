module DifferentiableBackwardEulerMooncakeExt

using DifferentiableBackwardEuler: dae_step, dae_steps, dae_steps_diffeq
using Mooncake
using Mooncake: CoDual, NoRData, NoFData, primal, tangent, zero_tangent,
                @is_primitive, DefaultCtx
import LinearAlgebra

# ======================================================================
# dae_step — native rrule!!
#
# Handles sparse Jacobians (f_y, f_p returning SparseMatrixCSC) correctly
# because array gradients are accumulated in-place via .+= into the input
# tangents, and Vector .+= SparseVector works through Julia broadcasting.
# ======================================================================

# --- positional args (no kwargs) ---

@is_primitive DefaultCtx Tuple{
    typeof(dae_step),
    AbstractVector{<:Real}, Real,
    Any, Any, Any, Any,
    AbstractVector{<:Real}, Real, Any,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(dae_step)},
    y0_cd::CoDual, deltah_cd::CoDual,
    f_cd::CoDual, f_y_cd::CoDual, f_p_cd::CoDual, f_t_cd::CoDual,
    p_cd::CoDual, t_cd::CoDual, M_cd::CoDual,
)
    y0      = primal(y0_cd)
    deltah  = primal(deltah_cd)
    f       = primal(f_cd)
    f_y     = primal(f_y_cd)
    f_p     = primal(f_p_cd)
    f_t     = primal(f_t_cd)
    p       = primal(p_cd)
    t       = primal(t_cd)
    M       = primal(M_cd)

    y1 = dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, M)
    dy1 = zero_tangent(y1)

    function dae_step_pb(::NoRData)
        Jt = LinearAlgebra.transpose(M - deltah * f_y(y1, p, t))
        lambda = Jt \ dy1
        # Array gradients: accumulate into input tangents in-place.
        # Works for sparse f_p because Vector .+= SparseVector broadcasts correctly.
        tangent(y0_cd) .+= LinearAlgebra.transpose(M) * lambda
        tangent(p_cd)  .+= LinearAlgebra.transpose(deltah * f_p(y1, p, t)) * lambda
        # Scalar gradients: returned as rdata
        ddeltah = LinearAlgebra.dot(f(y1, p, t), lambda)
        dt      = LinearAlgebra.dot(deltah * f_t(y1, p, t), lambda)
        return (NoRData(), NoRData(), ddeltah,
                NoRData(), NoRData(), NoRData(), NoRData(),
                NoRData(), dt, NoRData())
    end
    return CoDual(y1, dy1), dae_step_pb
end

# --- with kwargs via Core.kwcall ---

@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(dae_step),
    AbstractVector{<:Real}, Real,
    Any, Any, Any, Any,
    AbstractVector{<:Real}, Real, Any,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    ::CoDual{typeof(dae_step)},
    y0_cd::CoDual, deltah_cd::CoDual,
    f_cd::CoDual, f_y_cd::CoDual, f_p_cd::CoDual, f_t_cd::CoDual,
    p_cd::CoDual, t_cd::CoDual, M_cd::CoDual,
)
    kwargs  = primal(kwargs_cd)
    y0      = primal(y0_cd)
    deltah  = primal(deltah_cd)
    f       = primal(f_cd)
    f_y     = primal(f_y_cd)
    f_p     = primal(f_p_cd)
    f_t     = primal(f_t_cd)
    p       = primal(p_cd)
    t       = primal(t_cd)
    M       = primal(M_cd)

    y1 = dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, M; kwargs...)
    dy1 = zero_tangent(y1)

    function dae_step_kw_pb(::NoRData)
        Jt = LinearAlgebra.transpose(M - deltah * f_y(y1, p, t))
        lambda = Jt \ dy1
        tangent(y0_cd) .+= LinearAlgebra.transpose(M) * lambda
        tangent(p_cd)  .+= LinearAlgebra.transpose(deltah * f_p(y1, p, t)) * lambda
        ddeltah = LinearAlgebra.dot(f(y1, p, t), lambda)
        dt      = LinearAlgebra.dot(deltah * f_t(y1, p, t), lambda)
        return (NoRData(), NoRData(), NoRData(),
                NoRData(), ddeltah,
                NoRData(), NoRData(), NoRData(), NoRData(),
                NoRData(), dt, NoRData())
    end
    return CoDual(y1, dy1), dae_step_kw_pb
end

# ======================================================================
# dae_steps(t0, tfinal) — native rrule!!
#
# This variant delegates to DifferentialEquations.solve for adaptive
# stepping, which Mooncake cannot differentiate through. We declare it
# as a primitive and compute the adjoint manually.
#
# dae_steps(ts) is NOT declared as a primitive — Mooncake differentiates
# through its loop body, composing the dae_step rrule!! at each step.
# ======================================================================

# --- positional args (no kwargs) ---

@is_primitive DefaultCtx Tuple{
    typeof(dae_steps),
    AbstractVector{<:Real},
    Any, Any, Any, Any,
    AbstractVector{<:Real},
    Real, Real, Any,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(dae_steps)},
    y0_cd::CoDual,
    f_cd::CoDual, f_y_cd::CoDual, f_p_cd::CoDual, f_t_cd::CoDual,
    p_cd::CoDual,
    t0_cd::CoDual, tfinal_cd::CoDual, M_cd::CoDual,
)
    y0      = primal(y0_cd)
    f       = primal(f_cd)
    f_y     = primal(f_y_cd)
    f_p     = primal(f_p_cd)
    f_t     = primal(f_t_cd)
    p       = primal(p_cd)
    t0      = primal(t0_cd)
    tfinal  = primal(tfinal_cd)
    M       = primal(M_cd)

    soln = dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, M)
    y    = Matrix(soln)
    ts   = soln.t

    dy = zero_tangent(y)

    function dae_steps_t0tf_pb(::NoRData)
        n = size(y, 2)
        grad = copy(view(dy, :, n))
        for i = n:-1:2
            dt_i   = ts[i] - ts[i-1]
            y1_i   = y[:, i]
            Jt     = LinearAlgebra.transpose(M - dt_i * f_y(y1_i, p, ts[i]))
            lambda = Jt \ grad
            tangent(p_cd) .+= LinearAlgebra.transpose(dt_i * f_p(y1_i, p, ts[i])) * lambda
            propagated = LinearAlgebra.transpose(M) * lambda
            if i > 2
                grad = propagated .+ view(dy, :, i - 1)
            else
                tangent(y0_cd) .+= propagated .+ view(dy, :, 1)
            end
        end
        return (NoRData(),           # dae_steps
                NoRData(),           # y0 (array, gradient via fdata)
                NoRData(), NoRData(), NoRData(), NoRData(),  # f through f_t
                NoRData(),           # p (array, gradient via fdata)
                zero(t0), zero(tfinal),  # t0, tfinal (Float64)
                NoRData())           # M
    end
    return CoDual(y, dy), dae_steps_t0tf_pb
end

# --- with kwargs via Core.kwcall ---

@is_primitive DefaultCtx Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(dae_steps),
    AbstractVector{<:Real},
    Any, Any, Any, Any,
    AbstractVector{<:Real},
    Real, Real, Any,
}

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kwargs_cd::CoDual{<:NamedTuple},
    ::CoDual{typeof(dae_steps)},
    y0_cd::CoDual,
    f_cd::CoDual, f_y_cd::CoDual, f_p_cd::CoDual, f_t_cd::CoDual,
    p_cd::CoDual,
    t0_cd::CoDual, tfinal_cd::CoDual, M_cd::CoDual,
)
    kwargs  = primal(kwargs_cd)
    y0      = primal(y0_cd)
    f       = primal(f_cd)
    f_y     = primal(f_y_cd)
    f_p     = primal(f_p_cd)
    f_t     = primal(f_t_cd)
    p       = primal(p_cd)
    t0      = primal(t0_cd)
    tfinal  = primal(tfinal_cd)
    M       = primal(M_cd)

    soln = dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, M; kwargs...)
    y    = Matrix(soln)
    ts   = soln.t

    dy = zero_tangent(y)

    function dae_steps_t0tf_kw_pb(::NoRData)
        n = size(y, 2)
        grad = copy(view(dy, :, n))
        for i = n:-1:2
            dt_i   = ts[i] - ts[i-1]
            y1_i   = y[:, i]
            Jt     = LinearAlgebra.transpose(M - dt_i * f_y(y1_i, p, ts[i]))
            lambda = Jt \ grad
            tangent(p_cd) .+= LinearAlgebra.transpose(dt_i * f_p(y1_i, p, ts[i])) * lambda
            propagated = LinearAlgebra.transpose(M) * lambda
            if i > 2
                grad = propagated .+ view(dy, :, i - 1)
            else
                tangent(y0_cd) .+= propagated .+ view(dy, :, 1)
            end
        end
        return (NoRData(), NoRData(), NoRData(),  # kwcall, kwargs, dae_steps
                NoRData(),                          # y0
                NoRData(), NoRData(), NoRData(), NoRData(),  # f through f_t
                NoRData(),                          # p
                zero(t0), zero(tfinal),             # t0, tfinal
                NoRData())                          # M
    end
    return CoDual(y, dy), dae_steps_t0tf_kw_pb
end

end # module
