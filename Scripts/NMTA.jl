#=
NMTA:
- Julia version: 1.7.1
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie, (some edits, additions and bug fixes by Matthew Dicks)
- Function: Optimisation heursitic for agent-based models
- Structure:
    1. Structures
    2. Nelder-Mead
    3. Trace
    4. Optimization
    5. API
- Example:
    objective = NonDifferentiable(x -> WeightedSumofSquaredErrors(Parameters(Nᴸₜ = Int(abs(ceil(x[1]))), Nᴸᵥ = Int(abs(ceil(x[2]))), δ = abs(x[3]), κ = abs(x[4]), ν = abs(x[5]), σᵥ = abs(x[6])), 5, W, empiricalmoments, empiricallogreturns, gateway), initialsolution)
    optimizationoptions = Options(show_trace = true, store_trace = true, trace_simplex = true, extended_trace = true, iterations = sum(ta_rounds), ξ = 0.15, ta_rounds = ta_rounds, f_reltol = f_reltol)
    @time result = !isnothing(neldermeadstate) ? Optimize(objective, initialsolution, optimizationoptions, neldermeadstate) : Optimize(objective, initialsolution, optimizationoptions)
=#
using Printf
import StatsBase: var
import LinearAlgebra: rmul!
import Random: rand
import NLSolversBase: value, value!, value!!, NonDifferentiable
import Distributions: Normal
#---------------------------------------------------------------------------------------------------

#----- Structures -----#
struct Options{T <: Real}
    f_reltol::Vector{Float64}
    g_abstol::T
    iterations::Int64
    store_trace::Bool
    trace_simplex::Bool
    show_trace::Bool
    extended_trace::Bool
    show_every::Int64
    time_limit::Float64
    ξ::Float64
    ta_rounds::Vector{Int64}
end
function Options(; f_reltol::Vector{Float64} = Vector{Float64}(), ta_rounds::Vector{Int64} = Vector{Int64}(), g_abstol::Real = 1e-8, iterations::Int64 = 1_000, store_trace::Bool = false, trace_simplex::Bool = false, show_trace::Bool = false, extended_trace::Bool = false, show_every::Int64 = 1, time_limit = NaN, ξ = 0.0)
    Options(f_reltol, g_abstol, iterations, store_trace, trace_simplex, show_trace, extended_trace, show_every, Float64(time_limit), ξ, ta_rounds)
end
struct OptimizationState{Tf <: Real}
    iteration::Int64
    value::Tf
    g_norm::Tf
    metadata::Dict
end
const OptimizationTrace{Tf} = Vector{OptimizationState{Tf}}
abstract type Simplexer end
struct AffineSimplexer <: Simplexer
    a::Float64
    b::Float64
    AffineSimplexer(; a = 0.025, b = 0.5) = new(a, b)
end
abstract type NMParameters end
struct AdaptiveParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    AdaptiveParameters(; α = 1.0, β = 1.0, γ = 0.75 , δ = 1.0) = new(α, β, γ, δ)
end
struct FixedParameters <: NMParameters
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    FixedParameters(; α = 1.0, β = 2.0, γ = 0.5, δ = 0.5) = new(α, β, γ, δ)
end
parameters(P::AdaptiveParameters, n::Integer) = (P.α, P.β + 2/n, P.γ - 1/2n, P.δ - 1/n)
parameters(P::FixedParameters, n::Integer) = (P.α, P.β, P.γ, P.δ)
struct NelderMead{Ts <: Simplexer, Tp <: NMParameters}
    initial_simplex::Ts
    parameters::Tp
end
function NelderMead(; kwargs...)
    KW = Dict(kwargs)
    if haskey(KW, :initial_simplex) || haskey(KW, :parameters)
        initial_simplex, parameters = AffineSimplexer(), AdaptiveParameters()
        haskey(KW, :initial_simplex) && (initial_simplex = KW[:initial_simplex])
        haskey(KW, :parameters) && (parameters = KW[:parameters])
        return NelderMead(initial_simplex, parameters)
    else
        return NelderMead(AffineSimplexer(), AdaptiveParameters())
    end
end
mutable struct NelderMeadState{Tx <: AbstractArray, T <: Real, Tfs <: AbstractArray}
    x::Tx
    m::Int
    simplex::Vector{Tx}
    x_centroid::Tx
    x_lowest::Tx
    x_second_highest::Tx
    x_highest::Tx
    x_reflect::Tx
    x_cache::Tx
    f_simplex::Tfs
    nm_x::T
    f_lowest::T
    i_order::Vector{Int}
    α::T
    β::T
    γ::T
    δ::T
    step_type::String
    iteration::Int64
end
mutable struct OptimizationResults{Tx <: AbstractArray, Tf <: Real}
    initial_x::Tx
    minimizer::Tx
    minimum::Tf
    iterations::Int64
    iteration_converged::Bool
    f_reltol::Vector{Float64}
    g_converged::Bool
    g_abstol::Float64
    f_increased::Bool
    trace::OptimizationTrace{Tf}
    time_limit::Float64
    time_run::Float64
    stopped_by_time_limit::Bool
    end_state::NelderMeadState
end
#---------------------------------------------------------------------------------------------------

#----- Process Point -----#
function ProcessPoint(simplexPoint::Vector{Float64}, NᴸₜMax::Int64 = 20, NᴸᵥMax::Int64 = 20, δMax::Float64 = 10.0, σᵥMax::Float64 = 0.05)
    simplexPoint[1] = min(ceil(abs(simplexPoint[1])),NᴸₜMax)
    simplexPoint[2] = min(ceil(abs(simplexPoint[2])),NᴸᵥMax)
    simplexPoint[3] = min(abs(simplexPoint[3]),δMax)
    simplexPoint[4] = abs(simplexPoint[4])
    simplexPoint[5] = max(abs(simplexPoint[5]),1.5) # make sure ν is always greater than 1
    simplexPoint[end] = min(abs(simplexPoint[end]),σᵥMax)
    if simplexPoint[4] == 0.0 # ensure κ != 0
        simplexPoint[4] = 0.001
    end
    return simplexPoint
end
#---------------------------------------------------------------------------------------------------

#----- Nelder-Mead -----#
NelderMeadObjective(y::Vector, m::Integer, n::Integer) = sqrt(var(y) * (m / n))
function simplexer(S::AffineSimplexer, initial_x::Tx) where Tx <: AbstractArray
    n = length(initial_x)
    initial_simplex = Tx[copy(initial_x) for i = 1:(n + 1)]
    for j ∈ eachindex(initial_x)
        initial_simplex[j + 1][j] = (1 + S.b) * initial_simplex[j + 1][j] + S.a
    end
    initial_simplex
end
function centroid!(c::AbstractArray{T}, simplex, h::Integer = 0) where T # centroid except h-th vertex
    n = length(c)
    fill!(c, zero(T))
    for i in eachindex(simplex)
        if i != h
            xi = simplex[i]
            c .+= xi
        end
    end
    rmul!(c, T(1)/n)
end
centroid(simplex, h::Integer) = centroid!(similar(simplex[1]), simplex, h)
function InitialState(nelder_mead::NelderMead, f::NonDifferentiable, initial_x::Tx) where Tx <: AbstractArray
    T = eltype(initial_x)
    n = length(initial_x)
    m = n + 1
    simplex = [ProcessPoint(p) for p in simplexer(nelder_mead.initial_simplex, initial_x)]
    f_simplex = zeros(T, m)
    value!!(f, first(simplex))
    f_simplex[1] = value(f)
    for i in 2:length(simplex)
        f_simplex[i] = value(f, simplex[i])
    end
    i_order = sortperm(f_simplex) # Get the indices that correspond to the ordering of the f values at the vertices. i_order[1] is the index in the simplex of the vertex with the lowest function value, and i_order[end] is the index in the simplex of the vertex with the highest function value
    α, β, γ, δ = parameters(nelder_mead.parameters, n)
    NelderMeadState(copy(initial_x), # Variable to hold final minimizer value for MultivariateOptimizationResults
        m, # Number of vertices in the simplex
        simplex, # Maintain simplex in state.simplex
        centroid(simplex,  i_order[m]), # Maintain centroid in state.centroid
        copy(initial_x), # Store cache in state.x_lowest
        copy(initial_x), # Store cache in state.x_second_highest
        copy(initial_x), # Store cache in state.x_highest
        copy(initial_x), # Store cache in state.x_reflect
        copy(initial_x), # Store cache in state.x_cache
        f_simplex, # Store objective values at the vertices in state.f_simplex
        T(NelderMeadObjective(f_simplex, n, m)), # Store NelderMeadObjective in state.nm_x
        f_simplex[i_order[1]], # Store lowest f in state.f_lowest
        i_order, # Store a vector of rankings of objective values
        T(α), T(β), T(γ), T(δ), "initial", 0)
end
function ThresholdAccepting!(f::NonDifferentiable, state::NelderMeadState, τ::Float64)
    n, m = length(state.x), state.m
    paramindex = rand(1:n)
    perturbation = zeros(Float64, n)
    perturbation[paramindex] = rand(Normal(0, abs(sum(getindex.(state.simplex, paramindex)) / 2m)))
    state.x_cache = ProcessPoint(state.simplex[state.i_order[1]] .+ perturbation)
    f_perturb = value(f, state.x_cache)
    if f_perturb < state.f_simplex[state.i_order[1]] + τ # Only accept if perturbation is now better than the best + some threshold
        # Update state
        copyto!(state.simplex[state.i_order[1]], state.x_cache) # Replace solution with new solution
        @inbounds state.f_simplex[state.i_order[1]] = f_perturb # Replace objective value with new value
    end
    state.step_type = "thresholding"
    sortperm!(state.i_order, state.f_simplex) # Sort indeces of simplexes in ascending order of objective value
    state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
end
function SimplexSearch!(f::NonDifferentiable, state::NelderMeadState, τ::Float64)
    shrink = false; n, m = length(state.x), state.m # Augment the iteration counter
    centroid!(state.x_centroid, state.simplex, state.i_order[m])
    copyto!(state.x_lowest, state.simplex[state.i_order[1]])
    copyto!(state.x_second_highest, state.simplex[state.i_order[n]])
    copyto!(state.x_highest, state.simplex[state.i_order[m]])
    state.f_lowest = state.f_simplex[state.i_order[1]]
    f_second_highest = state.f_simplex[state.i_order[n]]
    f_highest = state.f_simplex[state.i_order[m]]
    # Compute a reflection
    state.x_reflect = ProcessPoint(state.x_centroid .+ state.α .* (state.x_centroid .- state.x_highest))
    f_reflect = value(f, state.x_reflect)
    if f_reflect < state.f_lowest + τ # Reflection has improved the objective
        # Compute an expansion
        state.x_cache = ProcessPoint(state.x_centroid .+ state.β .* (state.x_reflect .- state.x_centroid))
        f_expand = value(f, state.x_cache)
        if f_expand < f_reflect + τ # Expansion has improved the objective
            # Update state
            copyto!(state.simplex[state.i_order[m]], state.x_cache)
            @inbounds state.f_simplex[state.i_order[m]] = f_expand
            state.step_type = "expansion"
        else # Expansion did not improve the objective
            # Update state
            copyto!(state.simplex[state.i_order[m]], state.x_reflect)
            @inbounds state.f_simplex[state.i_order[m]] = f_reflect
            state.step_type = "reflection"
        end
        # Shift all order indeces, and wrap the last one around to the first (update best objective)
        i_highest = state.i_order[m]
        @inbounds for i = m:-1:2
            state.i_order[i] = state.i_order[i-1]
        end
        state.i_order[1] = i_highest
        state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
    elseif f_reflect < f_second_highest + τ # Reflection is better than the second worst
        # Update state
        copyto!(state.simplex[state.i_order[m]], state.x_reflect)
        @inbounds state.f_simplex[state.i_order[m]] = f_reflect
        state.step_type = "reflection"
        sortperm!(state.i_order, state.f_simplex)
        state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
    else
        if f_reflect < f_highest + τ # Reflection is better than the worst but mot better than the second worst
            # Outside contraction
            state.x_cache = ProcessPoint(state.x_centroid .+ state.γ .* (state.x_reflect .- state.x_centroid))
            f_outside_contraction = value(f, state.x_cache)
            if f_outside_contraction < f_reflect + τ
                # Update state
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_outside_contraction
                state.step_type = "outside contraction"
                sortperm!(state.i_order, state.f_simplex)
                state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
            else
                shrink = true
            end
        else # f_reflect > f_highest - new worst
            # Inside constraction
            state.x_cache = ProcessPoint(state.x_centroid .- state.γ .* (state.x_reflect .- state.x_centroid))
            f_inside_contraction = value(f, state.x_cache)
            if f_inside_contraction < f_highest + τ
                # Update state
                copyto!(state.simplex[state.i_order[m]], state.x_cache)
                @inbounds state.f_simplex[state.i_order[m]] = f_inside_contraction
                state.step_type = "inside contraction"
                sortperm!(state.i_order, state.f_simplex)
                state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
            else
                shrink = true
            end
        end
    end
    if shrink # Apply shrinkage if the worst could not be improved
        for i = 2:m
            ord = state.i_order[i]
            # Update state
            copyto!(state.simplex[ord], ProcessPoint(state.x_lowest .+ state.δ*(state.simplex[ord] .- state.x_lowest)))
            state.f_simplex[ord] = value(f, state.simplex[ord])
        end
        state.step_type = "shrink"
        sortperm!(state.i_order, state.f_simplex)
        state.f_lowest = state.f_simplex[state.i_order[1]] # I added to ensure the f_lowest is always set to the correct value (needed to be updated)
    end
    state.nm_x = NelderMeadObjective(state.f_simplex, n, m)
end
function PostProcess!(f::NonDifferentiable, state::NelderMeadState)
    sortperm!(state.i_order, state.f_simplex)
    x_centroid_min = ProcessPoint(centroid(state.simplex, state.i_order[state.m]))
    f_centroid_min = value(f, x_centroid_min)
    f_min, i_f_min = findmin(state.f_simplex)
    x_min = state.simplex[i_f_min]
    if f_centroid_min < f_min
        x_min = x_centroid_min
        f_min = f_centroid_min
    end
    f.F = f_min
    state.x .= x_min
end
function PostProcessError!(f::NonDifferentiable, state::NelderMeadState) # if an error occurs in the simulation we just want to save state
    # get the current min value of the function
    f_min, i_f_min = findmin(state.f_simplex)

    # get the minimizer
    x_min = state.simplex[i_f_min]

    # update the final state
    f.F = f_min
    state.x .= x_min

end
# Convergence
function AssessConvergence(state::NelderMeadState, options::Options)
    g_converged = state.nm_x <= options.g_abstol # Stopping criterion
    return g_converged, false
end
function InitialConvergence(state::NelderMeadState, initial_x::AbstractArray, options::Options) #f::NonDifferentiable,
    nm_x = NelderMeadObjective(state.f_simplex, state.m, length(initial_x))
    nm_x <= options.g_abstol#, !isfinite(nmo)   !isfinite(value(f)) ||
end
#---------------------------------------------------------------------------------------------------

#----- Trace -----#
function Trace!(tr::OptimizationTrace, state::NelderMeadState, iteration::Int64, options::Options, curr_time = time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["centroid"] = copy(state.x_centroid)
        dt["step_type"] = state.step_type
    end
    if options.trace_simplex
        dt["simplex"] = [copy(simplex_vec) for simplex_vec in state.simplex] # I added copy here to ensure that a copy of the simplex is stored and wont get changed later
        dt["simplex_values"] = copy(state.f_simplex)
    end
    os = OptimizationState(iteration, state.f_lowest, state.nm_x, dt) # changed to make sure the current lowest is always stored
    if options.store_trace
        push!(tr, os)
    end
    if options.show_trace
        if iteration % options.show_every == 0
            show(os)
            flush(stdout)
        end
    end
end
#---------------------------------------------------------------------------------------------------

#----- Optimization -----# (assumes there will be no errors in the initialization)
function Optimize(f::NonDifferentiable{Tf, Tx}, initial_x::Tx, options::Options{T} = Options(), state::NelderMeadState = InitialState(NelderMead(), f, initial_x)) where {Tx <: AbstractArray, Tf <: Real, T <: Real}
    t₀ = time() # Initial time stamp used to control early stopping by options.time_limit
    tr = OptimizationTrace{Tf}() # Store optimization trace
    thresholds = !isempty(options.ta_rounds) ? reduce(vcat, fill.(options.f_reltol, options.ta_rounds)) : zeros(options.iterations)
    tracing = options.store_trace || options.show_trace || options.extended_trace
    stopped, stopped_by_time_limit, f_increased = false, false, false
    g_converged = InitialConvergence(state, initial_x, options) # Converged if criterion is met
    # iteration = 0 # Counter
    if options.show_trace # Print header
        @printf "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
        @printf "------   --------------    --------------\n"
    end
    t = time()
    Trace!(tr, state, state.iteration, options, t - t₀)
    start_time = now()
    while !g_converged && !stopped_by_time_limit && state.iteration < options.iterations
        state.iteration += 1
        println(string(thresholds[state.iteration], "      ", thresholds[state.iteration] * sum(state.f_simplex) / state.m))
        println(run(`free -m`))
        println("Iterations = ", state.iteration)
        sleep(1)
        if rand() < options.ξ
            try
                ThresholdAccepting!(f, state, thresholds[state.iteration] * (sum(state.f_simplex) / state.m))
            catch e
                println(e)
                @error "Something went wrong" exception=(e, catch_backtrace())
                PostProcessError!(f, state)
                return OptimizationResults{Tx, Tf}(initial_x, state.x, value(f), state.iteration, state.iteration == options.iterations, options.f_reltol, g_converged, Float64(options.g_abstol), f_increased, tr, options.time_limit, t - t₀, stopped_by_time_limit, state)
            end
        else
            try
                SimplexSearch!(f, state, thresholds[state.iteration] * (sum(state.f_simplex) / state.m)) # Percentage of best solution
            catch e
                println(e)
                @error "Something went wrong" exception=(e, catch_backtrace())
                PostProcessError!(f, state)
                return OptimizationResults{Tx, Tf}(initial_x, state.x, value(f), state.iteration, state.iteration == options.iterations, options.f_reltol, g_converged, Float64(options.g_abstol), f_increased, tr, options.time_limit, t - t₀, stopped_by_time_limit, state)
            end
        end
        g_converged, f_increased = AssessConvergence(state, options)
        if tracing
            Trace!(tr, state, state.iteration, options, time() - t₀)
        end
        t = time()
        stopped_by_time_limit = t - t₀ > options.time_limit
    end
    PostProcess!(f, state)
    println("While loop time = ", now() - start_time)
    return OptimizationResults{Tx, Tf}(initial_x, state.x, value(f), state.iteration, state.iteration == options.iterations, options.f_reltol, g_converged, Float64(options.g_abstol), f_increased, tr, options.time_limit, t - t₀, stopped_by_time_limit, state)
end
#---------------------------------------------------------------------------------------------------

#----- API -----#
minimizer(r::OptimizationResults) = r.minimizer
optimum(r::OptimizationResults) = r.minimum
iterations(r::OptimizationResults) = r.iterations
iteration_limit_reached(r::OptimizationResults) = r.iteration_converged
trace(r::OptimizationResults) = length(r.trace) > 0 ? r.trace : error("No trace in optimization results. To get a trace, run optimize() with store_trace = true.")
converged(r::OptimizationResults) = r.g_converged
f_reltol(r::OptimizationResults) = r.f_reltol
g_abstol(r::OptimizationResults) = r.g_abstol
initial_state(r::OptimizationResults) = r.initial_x
time_limit(r::OptimizationResults) = r.time_limit
time_run(r::OptimizationResults) = r.time_run
g_norm_trace(r::OptimizationResults) = [ state.g_norm for state in trace(r) ]
f_trace(r::OptimizationResults) = [ state.value for state in trace(r) ]
end_state(r::OptimizationResults) = r.end_state
function centroid_trace(r::OptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "centroid") && error("Trace does not contain centroid. To get a trace of the centroid, run optimize() with extended_trace = true")
    [ state.metadata["centroid"] for state in tr ]
end
function simplex_trace(r::OptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "simplex") && error("Trace does not contain simplex. To get a trace of the simplex, run optimize() with trace_simplex = true")
    [ state.metadata["simplex"] for state in tr ]
end
function simplex_value_trace(r::OptimizationResults)
    tr = trace(r)
    !haskey(tr[1].metadata, "simplex_values") && error("Trace does not contain objective values at the simplex. To get a trace of the simplex values, run optimize() with trace_simplex = true")
    [ state.metadata["simplex_values"] for state in tr ]
end
function Base.show(io::IO, r::OptimizationResults)
    take = Iterators.take
    failure_string = "failure"
    if iteration_limit_reached(r)
        failure_string *= " (reached maximum number of iterations)"
    end
    if time_run(r) > time_limit(r)
        failure_string *= " (exceeded time limit of $(time_limit(r)))"
    end
    @printf io " * Status: %s\n\n" converged(r) ? "success" : failure_string
    @printf io " * Final objective value:     %e\n" optimum(r)
    @printf io "\n"
    @printf io " * Convergence measures\n"
    @printf io "    √(Σ(yᵢ-ȳ)²)/n %s %.1e\n" converged(r) ? "≤" : "≰" g_abstol(r)
    @printf io "\n"
    @printf io " * Work counters\n"
    @printf io "    Seconds run:   %d  (vs limit %d)\n" time_run(r) isnan(time_limit(r)) ? Inf : time_limit(r)
    @printf io "    Iterations:    %d\n" iterations(r)
    return
end
function Base.show(io::IO, trace::OptimizationTrace{<:Real})
    @printf io "Iter     Function value    √(Σ(yᵢ-ȳ)²)/n \n"
    @printf io "------   --------------    --------------\n"
    for state in trace.states
        show(io, state)
    end
    return
end
function Base.show(io::IO, t::OptimizationState{<:Real})
    @printf io "%6d   %14e    %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end
#---------------------------------------------------------------------------------------------------