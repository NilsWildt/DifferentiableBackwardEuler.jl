using Test

function naivegradient(f, x0s...; delta::Float64=1e-8)
    f0 = f(x0s...)
    grad = map(x -> zeros(size(x)), x0s)
    for j = 1:length(x0s)
        x0 = copy(x0s[j])
        for i = 1:length(x0)
            x = copy(x0)
            x[i] += delta
            xs = (x0s[1:j-1]..., x, x0s[j+1:length(x0s)]...)
            grad[j][i] = (f(xs...) - f0) / delta
        end
    end
    return grad
end

@testset "ODE" begin let
    import DifferentiableBackwardEuler
    DBE = DifferentiableBackwardEuler
    using DifferentialEquations
    import LinearAlgebra
    import SparseArrays # kept for reference; tests use LinearAlgebra.Diagonal for dense
    import Zygote
    import Mooncake

    f(u, p, t) = f_u(u, p, t) * u
    f_u(u, p, t) = SparseArrays.spdiagm(0 => p)
    f_p(u, p, t) = SparseArrays.spdiagm(0 => u)
    f_t(u, p, t) = zero(u)

    #--- test the forward solver analytically ---
    n = 2
    p = collect(1.0:n)
    u0 = ones(n)
    us = [u0]
    numsteps = 10^3
    ts = range(0, 1; length=numsteps)
    for i = 1:length(ts)-1
        push!(us, DBE.step(us[end], ts[i+1] - ts[i], f, f_u, f_p, f_t, p, ts[i+1]))
    end
    @test isapprox(exp.(p), us[end]; atol=1e2 / numsteps)

    #--- test gradients involving step(...) with Zygote ---
    g(p, u0) = sum(DBE.step(u0, ts[2] - ts[1], f, f_u, f_p, f_t, p, ts[2]))
    dgdp_zygote = Zygote.gradient((p, u0) -> g(p, u0), p, u0)
    dgdp_naive = naivegradient((p, u0) -> g(p, u0), p, u0)
    @test isapprox(dgdp_zygote[1], dgdp_naive[1]; atol=1e-6, rtol=1e-6)
    @test isapprox(dgdp_zygote[2], dgdp_naive[2]; atol=1e-6, rtol=1e-6)

    #--- test gradients involving step(...) with Mooncake ---
    g_p(p) = sum(DBE.step(u0, ts[2] - ts[1], f, f_u, f_p, f_t, p, ts[2]))
    cache = Mooncake.prepare_gradient_cache(g_p, p)
    _, (_, dgdp_mooncake) = Mooncake.value_and_gradient!!(cache, g_p, p)
    @test isapprox(dgdp_mooncake, dgdp_naive[1]; atol=1e-6, rtol=1e-6)

    #--- test scalar gradients (deltah, t) of step(...) with Zygote and Mooncake ---
    # The right-hand side depends on t so that the t-adjoint is nonzero.
    ft(u, p, t) = p .* u .+ t .^ 2
    ft_u(u, p, t) = SparseArrays.spdiagm(0 => p)
    ft_p(u, p, t) = SparseArrays.spdiagm(0 => u)
    ft_t(u, p, t) = fill(2 * t, length(u))
    h0, t1 = 0.1, 0.3
    g_h(h) = sum(DBE.step(u0, h, ft, ft_u, ft_p, ft_t, p, t1))
    g_t(t) = sum(DBE.step(u0, h0, ft, ft_u, ft_p, ft_t, p, t))
    g_h_kw(h) = sum(DBE.dae_step(u0, h, ft, ft_u, ft_p, ft_t, p, t1, LinearAlgebra.I; dbe_solver=DBE.nlsolve_solver))
    g_t_kw(t) = sum(DBE.dae_step(u0, h0, ft, ft_u, ft_p, ft_t, p, t, LinearAlgebra.I; dbe_solver=DBE.nlsolve_solver))
    fd(g, x; delta=1e-6) = (g(x + delta) - g(x - delta)) / (2 * delta)
    for (g_s, x) in ((g_h, h0), (g_t, t1), (g_h_kw, h0), (g_t_kw, t1))
        dg_naive = fd(g_s, x)
        @test isapprox(Zygote.gradient(g_s, x)[1], dg_naive; rtol=1e-5)
        cache = Mooncake.prepare_gradient_cache(g_s, x)
        _, (_, dg_mooncake) = Mooncake.value_and_gradient!!(cache, g_s, x)
        @test isapprox(dg_mooncake, dg_naive; rtol=1e-5)
    end

    #--- test gradients involving steps(..., ts) with Zygote ---
    numsteps = 10^1
    ts = collect(range(0, 1e0; length=numsteps))
    g(p, u0) = sum(DBE.steps(u0, f, f_u, f_p, f_t, p, ts)[:, end])
    @test isapprox(sum(exp.(p * ts[end])), g(p, u0); atol=1e2 / numsteps, rtol=1e2 / numsteps)
    dgdp_zygote = Zygote.gradient((p, u0) -> g(p, u0), p, u0)
    dgdp_naive = naivegradient((p, u0) -> g(p, u0), p, u0)
    @test isapprox(dgdp_zygote[1], dgdp_naive[1]; atol=1e-4, rtol=1e-4)
    @test isapprox(dgdp_zygote[2], dgdp_naive[2]; atol=1e-4, rtol=1e-4)

    #--- test gradients involving steps(..., ts) with Mooncake ---
    g_p(p) = sum(DBE.steps(u0, f, f_u, f_p, f_t, p, ts)[:, end])
    cache = Mooncake.prepare_gradient_cache(g_p, p)
    _, (_, dgdp_mooncake) = Mooncake.value_and_gradient!!(cache, g_p, p)
    @test isapprox(dgdp_mooncake, dgdp_naive[1]; atol=1e-4, rtol=1e-4)

    #--- test the forward solver against DifferentialEquations.jl ---
    u0 = [1 / 2, 1 / 3]
    f(u, p, t) = f_u(u, p, t) * u
    f_u(u, p, t) = SparseArrays.spdiagm(0 => p)
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, ImplicitEuler())
    dbesol = DBE.steps(u0, f, f_u, nothing, nothing, p, sol.t)
    for i = 1:length(sol.t)
        @test isapprox(dbesol[:, i], sol[:, i])
    end

    #--- test forward runs involving steps(..., t0, final) ---
    dbesol = DBE.steps(u0, f, f_u, nothing, nothing, p, tspan[1], tspan[2])
    for i = 1:length(sol.t)
        @test isapprox(dbesol[:, i], sol[:, i])
    end

    #--- test gradients involving steps(..., t0, tfinal) with Zygote ---
    f_p(u, p, t) = SparseArrays.spdiagm(0 => u)
    f_t(u, p, t) = zero(u)
    t0 = 0.0
    tfinal = 1e0
    g(p, u0) = sum(DBE.steps(u0, f, f_u, f_p, f_t, p, t0, tfinal)[:, end])
    dgdp_zygote = Zygote.gradient(g, p, u0)
    dgdp_naive = naivegradient(g, p, u0; delta=1e-2)
    @test isapprox(dgdp_zygote[2], dgdp_naive[2]; atol=1e-4, rtol=1e-4)
    @test isapprox(dgdp_zygote[1], dgdp_naive[1]; atol=1e-2, rtol=1e-2)

    #--- test gradients involving steps(..., t0, tfinal) with Mooncake ---
    g_p(p) = sum(DBE.steps(u0, f, f_u, f_p, f_t, p, t0, tfinal)[:, end])
    cache = Mooncake.prepare_gradient_cache(g_p, p)
    _, (_, dgdp_mooncake) = Mooncake.value_and_gradient!!(cache, g_p, p)
    @test isapprox(dgdp_mooncake, dgdp_naive[1]; atol=1e-2, rtol=1e-2)

end end
