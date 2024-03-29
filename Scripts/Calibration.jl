#=
Calibration:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie, (some code was adapted from https://github.com/IvanJericevich/IJPCTG-ABMCoinTossX)
- Function: Perform the calibration and visualise the convergence
- Structure:
    1. Generate empirical and simulated returns
    2. Moving block bootstrap to estimate epirical and simulated covariance matrix
    3. Defnintion of objective function to be minimised
    4. Calibration function
    5. Plot objective convergence plots
    6. Plot the parameter trace plots
    7. Generate confidence intervals for calibrated params
    8. Generate confidence intervals for the empirical and simulated moments
- Examples:
    1. Calibration
        date = DateTime("2019-07-08")
        startTime = date + Hour(9) + Minute(1)
        endTime = date + Hour(16) + Minute(50) 
        empiricalLogReturns, empiricalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)
        ta_rounds_arg = [5, 10, 20, 30, 35]
        f_reltol_arg = [0.3, 0.2, 0.1, 0.05, 0]
        initialsolution = [5, 5, 0.1, 3.5, 5, 0.015]
        @time Calibrate(initialsolution, empiricalLogReturns.MicroPriceLogReturns, empiricalMoments["empericalMicroPriceMoments"], ta_rounds = ta_rounds_arg, f_reltol = f_reltol_arg) # , neldermeadstate = neldermeadstate) [takes about 8hrs]
    2. Visualisations
        stacktrace = load("../Data/Calibration/OptimizationResult.jld")["result"]
        PlotObjectiveConvergence(stacktrace)
        ParameterTracePlots(stacktrace)
    3. Confidence intervals
        ParameterConfidenceIntervals([8, 6, 0.125, 3.389, 7.221, 0.041])
        MomentConfidenceIntervals(startTime, endTime)
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

#----- Generate simuklated log-returns and emperical moments -----#
function GenerateSimulatedReturnsAndMoments(empiricalMidPriceLogReturns::Vector{Float64}, empiricalMicroPriceLogReturns::Vector{Float64})
    println("Generating returns and moments for simulated data")
    simData = CSV.File(string("../Data/CoinTossX/L1LOB.csv"), missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
    filter!(x -> !ismissing(x.MidPrice), empericalData); filter!(x -> !ismissing(x.MicroPrice), simData)
    midPriceLogReturns = diff(log.(simData.MidPrice))
    microPriceLogReturns = diff(log.(simData.MicroPrice))
    simulatedLogReturns = DataFrame(MidPriceLogReturns = midPriceLogReturns, MicroPriceLogReturns = microPriceLogReturns)
    simulatedMidPriceMoments = Moments(midPriceLogReturns, empiricalMidPriceLogReturns)
    simulatedMicroPriceMoments = Moments(microPriceLogReturns, empiricalMicroPriceLogReturns)
    simulatedMoments = Dict("simulatedMidPriceMoments" => simulatedMidPriceMoments, "simulatedMicroPriceMoments" => simulatedMicroPriceMoments)
    return simulatedLogReturns, simulatedMoments
end
#---------------------------------------------------------------------------------------------------

#----- Moving block bootstrap to estimate covariance matrix of empirical moments on JSE time-series -----#
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

#----- Moving block bootstrap to estimate covariance matrix of simulated moments on CoinTossX time-series -----#
function MovingBlockBootstrapSimulated(logreturns::Vector{Float64}, empiricallogreturns::Vector{Float64}, iterations::Int64 = 1000, windowsize::Int64 = 2000)
    bootstrapmoments = fill(0.0, (iterations, 8))
    @showprogress "Computing weight matrix..." for i in 1:iterations
        indeces = rand(1:(length(logreturns) - windowsize + 1), Int(ceil(length(logreturns)/windowsize)))
        bootstrapreturns = Vector{Float64}()
        for index in indeces
            append!(bootstrapreturns, logreturns[index:(index  + windowsize - 1)])
        end
        moments = Moments(bootstrapreturns[1:length(logreturns)], empiricallogreturns)
        # bootstrapmoments[i,:] = [moments.μ moments.σ moments.κ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        bootstrapmoments[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
    end
    bootstrapmoments_df = DataFrame(bootstrapmoments, Symbol.(["Mean","Std","KS","Hurst","GPH","ADF","GARCH","Hill"]))
    CSV.write("../Data/Calibration/SimulatedBootstrapMoments.csv", bootstrapmoments_df)
    W = inv(cov(bootstrapmoments))
    save("../Data/Calibration/SimulatedW.jld", "SimulatedW", W)
end
#---------------------------------------------------------------------------------------------------

#----- Moving block bootstrap to estimate covariance matrix of empirical moments on JSE mid-price time-series -----#
function PlotBoostrapMoments()
    bootstrapmoments_df = CSV.File(string("../Data/Calibration/BootstrapMoments.csv")) |> DataFrame
    color = :blue
    for name in names(bootstrapmoments_df)
        NormalDistribution = Distributions.fit(Normal, bootstrapmoments_df[:,Symbol(name)])
        distribution = histogram(bootstrapmoments_df[:,Symbol(name)], normalize = :pdf, fillcolor = color, linecolor = color, xlabel = name, ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)
        savefig(distribution, string("../Images/Calibration/MomentDistributions/" * name * "Distribution.pdf"))
    end
end
#---------------------------------------------------------------------------------------------------

#----- Objective function to be minimized -----#
function WeightedSumofSquaredErrors(parameters::Parameters, replications::Int64, W::Array{Float64, 2}, empiricalmoments::Moments, empiricallogreturns::Vector{Float64}, gateway::TradingGateway)
    errormatrix = fill(0.0, (replications, 8))
    for i in 1:replications
        # set RL parameters so that they don't do anything
        rlParameters = RLParameters(0, Dict(), Millisecond(0), Millisecond(0), 0, 0, 0, 0, 0, 0, 0, Dict(), DataFrame(), DataFrame(), "", 0.0, 0.0, 0)

        # set the parameters that dictate output
        print_and_plot = false                    # Print out useful info about sim and plot simulation time series info
        write_messages = false                    # Says whether or not the messages data must be written to a file
        write_volume_spread = false
        rlTraders = false                        # should always be false in this file
        rlTraining = false                       # should always be false in this file

        midprice, microprice = simulate(parameters, rlParameters, gateway, rlTraders, rlTraining, print_and_plot, write_messages, write_volume_spread, seed = i, iteration = 0)
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

#----- Calibrate with NMTA optimization -----# (assumes CoinTossX has started)
function Calibrate(initialsolution::Vector{Float64}, empiricallogreturns::Vector{Float64}, empiricalmoments::Moments; f_reltol::Vector{Float64} = [0.3, 0.2, 0.1, 0], ta_rounds::Vector{Int64} = [4, 3, 2, 1], neldermeadstate = nothing)
    # if nelder mead initial state is not nothing need to do some processing
    StartJVM(); gateway = Login(1, 1)
    try
        cd(path_to_files * "/Scripts") # change back to path to files  
        W = load("../Data/Calibration/W.jld")["W"]
        objective = NonDifferentiable(x -> WeightedSumofSquaredErrors(Parameters(Nᴸₜ = Int(abs(ceil(x[1]))), Nᴸᵥ = Int(abs(ceil(x[2]))), δ = abs(x[3]), κ = abs(x[4]), ν = abs(x[5]), σᵥ = abs(x[6])), 5, W, empiricalmoments, empiricallogreturns, gateway), initialsolution)
        optimizationoptions = Options(show_trace = true, store_trace = true, trace_simplex = true, extended_trace = true, iterations = sum(ta_rounds), ξ = 0.15, ta_rounds = ta_rounds, f_reltol = f_reltol)
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

#----- Create the W for empirical and simulated returns -----#
# MovingBlockBootstrap(empiricalLogReturns.MicroPriceLogReturns)
# MovingBlockBootstrapSimulated(simulatedLogReturns.MicroPriceLogReturns, empiricalLogReturns.MicroPriceLogReturns)
# PlotBoostrapMoments()
#---------------------------------------------------------------------------------------------------

#----- Generate the empirical returns and moments -----#
# make sure these are the same for the stylized facts and sensitivity analysis
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50) 

# empiricalLogReturns, empiricalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)
# simulatedLogReturns, simulatedMoments = GenerateSimulatedReturnsAndMoments(empiricalLogReturns.MidPriceLogReturns, empiricalLogReturns.MicroPriceLogReturns)
#---------------------------------------------------------------------------------------------------

#----- Calibrate -----#
# ta_rounds_arg = [5, 10, 20, 30, 35]
# f_reltol_arg = [0.3, 0.2, 0.1, 0.05, 0]
# initialsolution = [5, 5, 0.1, 3.5, 5, 0.015]
# @time Calibrate(initialsolution, empiricalLogReturns.MicroPriceLogReturns, empiricalMoments["empericalMicroPriceMoments"], ta_rounds = ta_rounds_arg, f_reltol = f_reltol_arg) # , neldermeadstate = neldermeadstate)
#---------------------------------------------------------------------------------------------------

#----- Validate optimization results -----#
function PlotObjectiveConvergence(stacktrace)

    iters = iterations(stacktrace) + 1
    f = zeros(Float64, iters); g_norm = zeros(Float64, iters); f_simplex = fill(0.0, iters, 7)#; centr = fill(0.0, length(stacktrace), 5); metadata = Vector{Dict{Any, Any}}()
    i = 1
    j = 1
    for s in trace(stacktrace)
        f[i] = s.value                         # vertex with the lowest value (lowest with a tolerence in the begining)
        g_norm[i] = s.g_norm                   # √(Σ(yᵢ-ȳ)²)/n 
        f_simplex[i, :] = transpose(s.metadata["simplex_values"])
        i += 1
        j += 1
    end
    # Objectives
    objectives = plot(1:iters, f, seriestype = :line, linecolor = :blue, label = "Weighted SSE objective", xlabel = "Iteration", ylabel = "Weighted SSE objective", legendfontsize = 5, fg_legend = :transparent, tickfontsize = 5, xaxis = false, xticks = false, legend = :bottomleft, guidefontsize = 7, yscale = :log10, minorticks = true, left_margin = 5Plots.mm, right_margin = 15Plots.mm)
    plot!(twinx(), 1:iters, g_norm, seriestype = :line, linecolor = :purple, label = "Convergence criterion", ylabel = "Convergence criterion", legend = :topright, legendfontsize = 5, fg_legend = :transparent, tickfontsize = 5, yscale = :log10, minorticks = true, guidefontsize = 7)
    savefig(objectives, "../Images/Calibration/ObjectiveConvergence/NMTAFitnessBestVertexOG.pdf")
    # Simplex values
    convergence = plot(1:iters, f_simplex, seriestype = :line, linecolor = [:blue :purple :green :orange :red :black :magenta], xlabel = "Iteration", ylabel = "Weighted SSE objective", legend = false, tickfontsize = 5, guidefontsize = 7, yscale = :log10, minorticks = true)
    savefig(convergence, "../Images/Calibration/ObjectiveConvergence/NMTAFitnessAllSimplexValuesOG.pdf")
end
# stacktrace = load("../Data/Calibration/OptimizationResult.jld")["result"]
# PlotObjectiveConvergence(stacktrace)
#---------------------------------------------------------------------------------------------------

#----- Parameter Confidence Intervals -----#
function ParameterConfidenceIntervals(calibratedParams::Vector{Float64})
    W = load("../Data/Calibration/W.jld")["W"]
    B = load("../Data/SensitivityAnalysis/B.jld")["B"]
    sigmas = sqrt.(diag(B * inv(W) * transpose(B)))
    upper = calibratedParams .+ (1.96 .* sigmas)
    lower = calibratedParams .- (1.96 .* sigmas)
    parameters = [("Nt", "Nᴸₜ"), ("Nv", "Nᴸᵥ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    df = DataFrame(Parameters = first.(parameters), CalibratedParameters = calibratedParams, Lower = lower, Upper = upper)
    CSV.write("../Data/Calibration/parameters.csv", df)
end

# ParameterConfidenceIntervals([8, 6, 0.125, 3.389, 7.221, 0.041])
#---------------------------------------------------------------------------------------------------

#----- Parameter Trace Plots -----#
function ParameterTracePlots(stacktrace)
    iters = iterations(stacktrace) + 1
    meanTrace = fill(0.0, iters, 6)
    upperTrace = fill(0.0, iters, 6)
    lowerTrace = fill(0.0, iters, 6)
    parameters = [("Nt", "Nᴸₜ"), ("Nv", "Nᴸᵥ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    c = [:blue :purple :green :orange :red :black :magenta]
    for (i,param) in enumerate(parameters)
        t = fill(0.0, iters, 7)
        for (j,s) in enumerate(simplex_trace(stacktrace))
            t[j,:] = transpose(hcat(s...))[:,i]
        end
        p = plot(1:iters, t, seriestype = :line, linestyle = :dash, linecolor = c[i], xlabel = "Iteration", ylabel = last(param), legend = false, tickfontsize = 5, guidefontsize = 7, minorticks = true)
        plot!(1:iters, transpose(hcat(centroid_trace(stacktrace)...))[:,i], seriestype = :line, linestyle = :solid, linecolor = c[i], linewidth = 2, legend = false)
        savefig(p, "../Images/Calibration/ParameterConvergence/ParameterConvergence" * first(param) * ".pdf")
    end

end

# stacktrace = load("../Data/Calibration/OptimizationResult.jld")["result"]
# ParameterTracePlots(stacktrace)
#---------------------------------------------------------------------------------------------------

#----- Moment Confidence Intervals -----# (need to do if you get another days data)
function MomentConfidenceIntervals(startTime::DateTime, endTime::DateTime)
    W = load("../Data/Calibration/W.jld")["W"]

    # Emperical Moments and confidence intervals using inv(W) (emperical covariance matrix)
    empiricalLogReturns, empiricalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)
    sigmas = sqrt.(diag(inv(W)))
    empirical = [empiricalMoments["empericalMicroPriceMoments"].μ, empiricalMoments["empericalMicroPriceMoments"].σ, empiricalMoments["empericalMicroPriceMoments"].ks, empiricalMoments["empericalMicroPriceMoments"].hurst, empiricalMoments["empericalMicroPriceMoments"].gph, empiricalMoments["empericalMicroPriceMoments"].adf, empiricalMoments["empericalMicroPriceMoments"].garch, empiricalMoments["empericalMicroPriceMoments"].hill]
    empiricalLower = empirical .- (1.96 .* sigmas)
    empiricalUpper = empirical .+ (1.96 .* sigmas)

    # Simulated moments and confidence intervals using simulated variance covariance matrix
    SimulatedW = load("../Data/Calibration/SimulatedW.jld")["SimulatedW"]
    simulatedLogReturns, simulatedMoments = GenerateSimulatedReturnsAndMoments(empiricalLogReturns.MidPriceLogReturns, empiricalLogReturns.MicroPriceLogReturns)
    sigmas = sqrt.(diag(inv(SimulatedW)))
    simulated = [simulatedMoments["simulatedMicroPriceMoments"].μ, simulatedMoments["simulatedMicroPriceMoments"].σ, simulatedMoments["simulatedMicroPriceMoments"].ks, simulatedMoments["simulatedMicroPriceMoments"].hurst, simulatedMoments["simulatedMicroPriceMoments"].gph, simulatedMoments["simulatedMicroPriceMoments"].adf, simulatedMoments["simulatedMicroPriceMoments"].garch, simulatedMoments["simulatedMicroPriceMoments"].hill]
    simulatedLower = simulated .- (1.96 .* sigmas)
    simulatedUpper = simulated .+ (1.96 .* sigmas)

    # write to file
    moments = ["Mean", "Std", "KS", "Hurst", "GPH", "ADF", "GARCH", "Hill"]
    df = DataFrame(Moments = moments, Empirical = empirical, EmpiricalLower = empiricalLower, EmpiricalUpper = empiricalUpper, Simulated = simulated, SimulatedLower = simulatedLower, SimulatedUpper = simulatedUpper)
    CSV.write("../Data/Calibration/moments.csv", df)
end

# # make sure these are the same for the stylized facts and sensitivity analysis
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50) 
# MomentConfidenceIntervals(startTime, endTime)
#---------------------------------------------------------------------------------------------------