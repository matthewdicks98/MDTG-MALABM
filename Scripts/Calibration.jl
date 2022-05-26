#=
Calibration:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie
- Function: Calibrate CoinTossX ABM simulated mid-prices to JSE mid-price data
- Structure:
    1. Moving block bootstrap to estimate covariance matrix of empirical moments on JSE mid-price time-series
    2. Objective function to be minimized
    3. Calibrate with NMTA optimization
    4. Visualisation
- Examples:
    midprice = CSV.File("Data/JSECleanedTAQNPN.csv", select = [:MidPrice], limit = 40000) |> Tables.matrix |> vec |> x -> filter(y -> !isnan(y), x)
    W = MovingBlockBootstrap(midprice, 500)
    Calibrate(initialsolution)
=#
using JLD, CSV, Plots, ProgressMeter, LinearAlgebra, StatsPlots, StatsBase, Distributions
import Statistics: cov
import Random.rand
import Logging
Logging.disable_logging(Logging.Warn) # for ties warning in the estimation of the KS statistic
path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/Scripts"
cd(path_to_files)
include("Moments.jl")
include("ReactiveABM.jl"); include("NMTA.jl")
#---------------------------------------------------------------------------------------------------

#----- Generate emperical log-returns and emperical moments -----#
function GenerateEmpericalReturnsAndMoments(startTime::DateTime, endTime::DateTime)
    println("Generating returns and moments for: " * Dates.format(startTime, "yyyy-mm-ddTHH:MM:SS") * " to " * Dates.format(endTime, "yyyy-mm-ddTHH:MM:SS"))
    empericalData = CSV.File(string("../Data/JSE/L1LOB.csv"), missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
    filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, empericalData)
    filter!(x -> !ismissing(x.MidPrice), empericalData); filter!(x -> !ismissing(x.MicroPrice), empericalData)
    midPriceLogReturns = diff(log.(empericalData.MidPrice))
    microPriceLogReturns = diff(log.(empericalData.MicroPrice))
    empericalLogReturns = DataFrame(MidPriceLogReturns = midPriceLogReturns, MicroPriceLogReturns = microPriceLogReturns)
    empericalMidPriceMoments = Moments(midPriceLogReturns, midPriceLogReturns)
    empericalMicroPriceMoments = Moments(microPriceLogReturns, microPriceLogReturns)
    empericalMoments = Dict("empericalMidPriceMoments" => empericalMidPriceMoments, "empericalMicroPriceMoments" => empericalMicroPriceMoments)
    return empericalLogReturns, empericalMoments
end
#---------------------------------------------------------------------------------------------------


#----- Moving block bootstrap to estimate covariance matrix of empirical moments on JSE mid-price time-series -----#
function MovingBlockBootstrap(logreturns::Vector{Float64}, iterations::Int64 = 1000, windowsize::Int64 = 2000)
    bootstrapmoments = fill(0.0, (iterations, 8))
    @showprogress "Computing weight matrix..." for i in 1:iterations
        indeces = rand(1:(length(logreturns) - windowsize + 1), Int(ceil(length(logreturns)/windowsize)))
        bootstrapreturns = Vector{Float64}()
        for index in indeces
            append!(bootstrapreturns, logreturns[index:(index  + windowsize - 1)])
        end
        moments = Moments(bootstrapreturns[1:length(logreturns)], logreturns)
        # bootstrapmoments[i,:] = [moments.μ moments.σ moments.κ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        bootstrapmoments[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
    end
    bootstrapmoments_df = DataFrame(bootstrapmoments, Symbol.(["Mean","Std","KS","Hurst","GPH","ADF","GARCH","Hill"]))
    CSV.write("../Data/Calibration/BootstrapMoments.csv", bootstrapmoments_df)
    W = inv(cov(bootstrapmoments))
    save("../Data/Calibration/W.jld", "W", W)
end
#---------------------------------------------------------------------------------------------------

#----- Moving block bootstrap to estimate covariance matrix of empirical moments on JSE mid-price time-series -----#
function PlotBoostrapMoments()
    bootstrapmoments_df = CSV.File(string("../Data/Calibration/BootstrapMoments.csv")) |> DataFrame
    color = :blue
    for name in names(bootstrapmoments_df)
        NormalDistribution = Distributions.fit(Normal, bootstrapmoments_df[:,Symbol(name)])
        distribution = histogram(bootstrapmoments_df[:,Symbol(name)], normalize = :pdf, fillcolor = color, linecolor = color, xlabel = name, ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)
        savefig(distribution, string("../Images/Calibration/" * name * "Distribution.pdf"))
    end
end
#---------------------------------------------------------------------------------------------------

#----- Objective function to be minimized -----#
function WeightedSumofSquaredErrors(parameters::Parameters, replications::Int64, W::Array{Float64, 2}, empiricalmoments::Moments, empiricallogreturns::Vector{Float64}, gateway::TradingGateway)
    errormatrix = fill(0.0, (replications, 8))
    for i in 1:replications
        midprice, microprice = simulate(parameters, gateway, false, false, seed = i)
        if !isempty(microprice)
            filter!(x -> !isnan(x), microprice)
            logreturns = diff(log.(microprice))
            try
                simulatedmoments = Moments(logreturns, empiricallogreturns)
                errormatrix[i, :] = [simulatedmoments.μ-empiricalmoments.μ simulatedmoments.σ-empiricalmoments.σ simulatedmoments.ks-empiricalmoments.ks simulatedmoments.hurst-empiricalmoments.hurst simulatedmoments.gph-empiricalmoments.gph simulatedmoments.adf-empiricalmoments.adf simulatedmoments.garch-empiricalmoments.garch simulatedmoments.hill-empiricalmoments.hill]
            catch e
                println(e)
                errormatrix[i, :] = errormatrix[i - 1, :]
            end
        else
            return Inf
        end
    end
    GC.gc() # Garbage collection
    errors = mean(errormatrix, dims = 1)
    return (errors * W * transpose(errors))[1]
end
#---------------------------------------------------------------------------------------------------

#----- Calibrate with NMTA optimization -----#
function Calibrate(initialsolution::Vector{Float64}, empiricallogreturns::Vector{Float64}, empiricalmoments::Moments; f_reltol::Vector{Float64} = [0.3, 0.2, 0.1, 0], ta_rounds::Vector{Int64} = [4, 3, 2, 1], neldermeadstate = nothing)
    # if nelder mead initial state is not nothing need to do some processing
    StartCoinTossX(build = false); sleep(20); 
    StartJVM(); gateway = Login(1, 1)
    try
        cd(path_to_files * "/Scripts") # change back to path to files
        W = load("../Data/Calibration/W.jld")["W"]
        # counter = Counter(0)                         # !isempty(ta_rounds) ? sum(ta_rounds) : 30, also set replications to 4
        objective = NonDifferentiable(x -> WeightedSumofSquaredErrors(Parameters(Nᴸₜ = Int(abs(ceil(x[1]))), Nᴸᵥ = Int(abs(ceil(x[2]))), δ = abs(x[3]), κ = abs(x[4]), ν = abs(x[5]), σᵥ = abs(x[6])), 5, W, empiricalmoments, empiricallogreturns, gateway), initialsolution)
        optimizationoptions = Options(show_trace = true, store_trace = true, trace_simplex = true, extended_trace = true, iterations = 20, ξ = 0.15, ta_rounds = ta_rounds, f_reltol = f_reltol)
        @time result = !isnothing(neldermeadstate) ? Optimize(objective, initialsolution, optimizationoptions, neldermeadstate) : Optimize(objective, initialsolution, optimizationoptions)
        save("../Data/Calibration/OptimizationResultTest.jld", "result", result)
        Logout(gateway); StopCoinTossX()
    catch e
        Logout(gateway); StopCoinTossX()
        save("../Data/Calibration/OptimizationResultTest.jld", "result", result)
        @error "Something went wrong" exception=(e, catch_backtrace())
    end
end
#---------------------------------------------------------------------------------------------------

# make sure these are the same for the stylized facts and sensitivity analysis
date = DateTime("2019-07-08")
startTime = date + Hour(9) + Minute(1)
endTime = date + Hour(16) + Minute(50) ###### Change to 16:50

# empiricalLogReturns, empiricalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)

# MovingBlockBootstrap(empiricalLogReturns.MicroPriceLogReturns)

# PlotBoostrapMoments()

# initialState = load("../Data/Calibration/OptimizationResult13.jld")["result"]
# neldermeadstate = initialState.end_state
# initialsolution = minimizer(initialState)

ta_rounds_arg = [5, 10, 20, 30, 35]
f_reltol_arg = [0.3, 0.2, 0.1, 0.05, 0]
initialsolution = [5, 5, 0.1, 3.5, 5, 0.015]
@time Calibrate(initialsolution, empiricalLogReturns.MicroPriceLogReturns, empiricalMoments["empericalMicroPriceMoments"], ta_rounds = ta_rounds_arg, f_reltol = f_reltol_arg) # , neldermeadstate = neldermeadstate)

# stacktrace = load("../Data/Calibration/OptimizationResultTest.jld")["result"]
# stacktrace1 = load("../Data/Calibration/OptimizationResult2.jld")["result"]
# for s in trace(stacktrace)
#     println(s)
# end
# println(stacktrace.iterations)

# println(initial_state(stacktrace))
# println(minimizer(stacktrace))
# println(optimum(stacktrace))

#----- Validate optimization results -----#
# stacktrace = load("../Data/Calibration/OptimizationResultTest.jld")["result"]
# iters = iterations(stacktrace) + 1

# f = zeros(Float64, iters); g_norm = zeros(Float64, iters); f_simplex = fill(0.0, iters, 7)#; centr = fill(0.0, length(stacktrace), 5); metadata = Vector{Dict{Any, Any}}()

# i = 1
# j = 1
# for s in trace(stacktrace)
#     f[i] = s.value                         # vertex with the lowest value (lowest with a tolerence in the begining)
#     g_norm[i] = s.g_norm                   # √(Σ(yᵢ-ȳ)²)/n 
#     f_simplex[i, :] = transpose(s.metadata["simplex_values"])
#     global i += 1
#     global j += 1
# end

# println(f)
# println(f_simplex)
# for s in trace(stacktrace1)
#     if i == iterations(stacktrace) + 2
#         global i += 1
#         continue
#     else
#         f[j] = s.value                         # vertex with the lowest value (lowest with a tolerence in the begining)
#         g_norm[j] = s.g_norm                   # √(Σ(yᵢ-ȳ)²)/n 
#         f_simplex[j, :] = transpose(s.metadata["simplex_values"])
#         global i += 1
#         global j += 1
#     end
# end

# # Objectives
# objectives = plot(1:iters, f, seriestype = :line, linecolor = :blue, label = "Weighted SSE objective", xlabel = "Iteration", ylabel = "Weighted SSE objective", legendfontsize = 5, fg_legend = :transparent, tickfontsize = 5, xaxis = false, xticks = false, legend = :bottomleft, guidefontsize = 7, yscale = :log10, minorticks = true, left_margin = 5Plots.mm, right_margin = 15Plots.mm)
# plot!(twinx(), 1:iters, g_norm, seriestype = :line, linecolor = :purple, label = "Convergence criterion", ylabel = "Convergence criterion", legend = :topright, legendfontsize = 5, fg_legend = :transparent, tickfontsize = 5, yscale = :log10, minorticks = true, guidefontsize = 7)
# savefig(objectives, "../Images/Calibration/NMTAFitnessBestVertex10.pdf")
# Simplex values
# convergence = plot(1:iters, f_simplex, seriestype = :line, linecolor = [:blue :purple :green :orange :red :black :magenta], xlabel = "Iteration", ylabel = "Weighted SSE objective", legend = false, tickfontsize = 5, guidefontsize = 7, yscale = :log10, minorticks = true)
# savefig(convergence, "../Images/Calibration/NMTAFitnessAllSimplexValues10.pdf")
#---------------------------------------------------------------------------------------------------