ENV["JULIA_COPY_STACKS"]=1
using ProgressMeter, CSV, Plots, DataFrames, StatsPlots, Statistics, ColorSchemes, Dates, JLD, Combinatorics, Colors
using LinearAlgebra: diag, inv, transpose

# set working directory (the path to the Scripts/StylisedFacts.jl file)
path_to_folder = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/Scripts"
cd(path_to_folder)

include(path_to_folder * "/ReactiveABM.jl"); include(path_to_folder * "/CoinTossXUtilities.jl"); include(path_to_folder * "/Moments.jl") # This also includes CoinTossXUtilities.jl

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

#----- Sensitivity analysis -----#
function GenerateParameterCombinations(NᴸₜRange::Vector{Int64}, NᴸᵥRange::Vector{Int64}, δRange::Vector{Float64}, κRange::Vector{Float64}, νRange::Vector{Float64}, σᵥRange::Vector{Float64})
    println("Generating parameter combinations")
    parameterCombinations = Vector{Parameters}()
    for Nᴸₜ in NᴸₜRange
        for Nᴸᵥ in NᴸᵥRange
            for δ in δRange
                for κ in κRange
                    for ν in νRange
                        for σᵥ in σᵥRange
                            parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = 30, δ = δ, κ = κ, ν = ν, m₀ = 10000, σᵥ = σᵥ, λmin = 0.0005, λmax = 0.05, γ = Millisecond(1000), T = Millisecond(25000))
                            push!(parameterCombinations, parameters)
                        end
                    end
                end
            end
        end
    end
    return parameterCombinations
end
#---------------------------------------------------------------------------------------------------

#----- Sensitivity analysis -----# (ensure CoinTossX has started)
function SensitivityAnalysis(empericalLogReturns::DataFrame, empericalMoments::Dict, parameterCombinations::Vector{Parameters}, parameterCombinationsRange::Vector{Int64})
    # StartCoinTossX(false, false)
    StartJVM()
    gateway = Login(1, 1)
    open("../Data/SensitivityAnalysis/SensitivityAnalysisResults.csv", "w") do file
        println(file, "Type,Nt,Nv,Nh,Delta,Kappa,Nu,M0,SigmaV,LambdaMin,LambdaMax,Gamma,T,Seed,Mean,Std,Kurtosis,KS,Hurst,GPH,ADF,GARCH,Hill")
        for (i, parameters) in enumerate(parameterCombinations) # [parameterCombinationsRange[1]:parameterCombinationsRange[2]])
            try 
                seed = 1
                @time midPrices, microPrices = simulate(parameters, gateway, false, false, false, seed = seed)
                if isnothing(midPrices) && isnothing(microPrices)
                    println("\nParameter Set: $(i-1) finished\n")
                    break
                end
                println("\nParameter Set: $(i)\n")
                println(run(`free -m`))
                filter!(x -> !ismissing(x) && !(isnan(x)), midPrices); filter!(x -> !ismissing(x) && !(isnan(x)), microPrices)
                midPriceLogReturns = diff(log.(midPrices))
                microPriceLogReturns = diff(log.(microPrices))
                simulatedMidPriceMoments = Moments(midPriceLogReturns, empericalLogReturns.MidPriceLogReturns)
                simulatedMicroPriceMoments = Moments(microPriceLogReturns, empericalLogReturns.MicroPriceLogReturns)
                println(file, "MidPrice,", parameters.Nᴸₜ, ",", parameters.Nᴸᵥ, ",", parameters.Nᴴ, ",", parameters.δ, ",", parameters.κ, ",", parameters.ν, ",", parameters.m₀, ",", parameters.σᵥ, ",", parameters.λmin, ",", parameters.λmax, ",", parameters.γ, ",", parameters.T, ",", seed, ",", simulatedMidPriceMoments.μ, ",", simulatedMidPriceMoments.σ, ",", simulatedMidPriceMoments.κ, ",", simulatedMidPriceMoments.ks, ",", simulatedMidPriceMoments.hurst, ",", simulatedMidPriceMoments.gph, ",", simulatedMidPriceMoments.adf, ",", simulatedMidPriceMoments.garch, ",", simulatedMidPriceMoments.hill)
                println(file, "MicroPrice,", parameters.Nᴸₜ, ",", parameters.Nᴸᵥ, ",", parameters.Nᴴ, ",", parameters.δ, ",", parameters.κ, ",", parameters.ν, ",", parameters.m₀, ",", parameters.σᵥ, ",", parameters.λmin, ",", parameters.λmax, ",", parameters.γ, ",", parameters.T, ",", seed, ",", simulatedMicroPriceMoments.μ, ",", simulatedMicroPriceMoments.σ, ",", simulatedMicroPriceMoments.κ, ",", simulatedMicroPriceMoments.ks, ",", simulatedMicroPriceMoments.hurst, ",", simulatedMicroPriceMoments.gph, ",", simulatedMicroPriceMoments.adf, ",", simulatedMicroPriceMoments.garch, ",", simulatedMicroPriceMoments.hill)
                GC.gc()             # perform garbage collection
                
            catch e
                println(e)
                println(file, "MidPrice,", parameters.Nᴸₜ, ",", parameters.Nᴸᵥ, ",", parameters.Nᴴ, ",", parameters.δ, ",", parameters.κ, ",", parameters.ν, ",", parameters.m₀, ",", parameters.σᵥ, ",", parameters.λmin, ",", parameters.λmax, ",", parameters.γ, ",", parameters.T, ",", seed, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN)
                println(file, "MicroPrice,", parameters.Nᴸₜ, ",", parameters.Nᴸᵥ, ",", parameters.Nᴴ, ",", parameters.δ, ",", parameters.κ, ",", parameters.ν, ",", parameters.m₀, ",", parameters.σᵥ, ",", parameters.λmin, ",", parameters.λmax, ",", parameters.γ, ",", parameters.T, ",", seed, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN, ",", NaN)
            end
        end 
    end
    Logout(gateway)
    # StopCoinTossX()
end
#---------------------------------------------------------------------------------------------------

# collect the comand line arguments
# parameterCombinationsRange = map(x -> parse(Int64, x), ARGS)

# make sure these are the same for the stylized facts and Calibration
date = DateTime("2019-07-08")
startTime = date + Hour(9) + Minute(1)
endTime = date + Hour(16) + Minute(50) # Hour(17) ###### Change to 16:50

# empericalLogReturns, empericalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)

# NᴸₜRange = [3,6,9,12]
# NᴸᵥRange = [3,6,9,12]
# δRange = collect(range(0.01, 0.2, length = 4))
# κRange = collect(range(2, 5, length = 4))
# νRange = collect(range(2, 8, length = 4))
# σᵥRange = collect(range(0.0025, 0.025, length = 4))

# parameterCombinations = GenerateParameterCombinations(NᴸₜRange, NᴸᵥRange, δRange, κRange, νRange, σᵥRange)

# @time SensitivityAnalysis(empericalLogReturns, empericalMoments, parameterCombinations, parameterCombinationsRange)

#----- Visualizations -----#

#----- Compute Objective Function -----#
function ComputeObjective(empericalMoments::Dict)
    W = load("../Data/Calibration/W.jld")["W"]
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResults.csv")) |> DataFrame
    objs = Vector{Float64}()
    for i in 1:nrow(sr)
        if i % 2 == 0
            # errors = [sr[i,:Mean]-empericalMoments["empericalMicroPriceMoments"].μ sr[i,:Std]-empericalMoments["empericalMicroPriceMoments"].σ sr[i,:Kurtosis]-empericalMoments["empericalMicroPriceMoments"].κ sr[i,:KS]-empericalMoments["empericalMicroPriceMoments"].ks sr[i,:Hurst]-empericalMoments["empericalMicroPriceMoments"].hurst sr[i,:GPH]-empericalMoments["empericalMicroPriceMoments"].gph sr[i,:ADF]-empericalMoments["empericalMicroPriceMoments"].adf sr[i,:GARCH]-empericalMoments["empericalMicroPriceMoments"].garch sr[i,:Hill]-empericalMoments["empericalMicroPriceMoments"].hill]
            errors = [sr[i,:Mean]-empericalMoments["empericalMicroPriceMoments"].μ sr[i,:Std]-empericalMoments["empericalMicroPriceMoments"].σ sr[i,:KS]-empericalMoments["empericalMicroPriceMoments"].ks sr[i,:Hurst]-empericalMoments["empericalMicroPriceMoments"].hurst sr[i,:GPH]-empericalMoments["empericalMicroPriceMoments"].gph sr[i,:ADF]-empericalMoments["empericalMicroPriceMoments"].adf sr[i,:GARCH]-empericalMoments["empericalMicroPriceMoments"].garch sr[i,:Hill]-empericalMoments["empericalMicroPriceMoments"].hill]
        else
            # errors = [sr[i,:Mean]-empericalMoments["empericalMidPriceMoments"].μ sr[i,:Std]-empericalMoments["empericalMidPriceMoments"].σ sr[i,:Kurtosis]-empericalMoments["empericalMidPriceMoments"].κ sr[i,:KS]-empericalMoments["empericalMidPriceMoments"].ks sr[i,:Hurst]-empericalMoments["empericalMidPriceMoments"].hurst sr[i,:GPH]-empericalMoments["empericalMidPriceMoments"].gph sr[i,:ADF]-empericalMoments["empericalMidPriceMoments"].adf sr[i,:GARCH]-empericalMoments["empericalMidPriceMoments"].garch sr[i,:Hill]-empericalMoments["empericalMidPriceMoments"].hill]
            errors = [sr[i,:Mean]-empericalMoments["empericalMidPriceMoments"].μ sr[i,:Std]-empericalMoments["empericalMidPriceMoments"].σ sr[i,:KS]-empericalMoments["empericalMidPriceMoments"].ks sr[i,:Hurst]-empericalMoments["empericalMidPriceMoments"].hurst sr[i,:GPH]-empericalMoments["empericalMidPriceMoments"].gph sr[i,:ADF]-empericalMoments["empericalMidPriceMoments"].adf sr[i,:GARCH]-empericalMoments["empericalMidPriceMoments"].garch sr[i,:Hill]-empericalMoments["empericalMidPriceMoments"].hill]
        end
        obj = errors * W * transpose(errors)
        push!(objs, obj[1])
    end
    insertcols!(sr, ncol(sr) + 1, "Objective" => objs)
    CSV.write("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv", sr)
end

# ComputeObjective(empericalMoments)
#---------------------------------------------------------------------------------------------------

#----- Moment Values For Parameter Marginals -----#
function Winsorize(paramvalues, momentvalues)
    df = DataFrame(ParamValues = paramvalues, MomentValues = momentvalues)
    upper = quantile(df.MomentValues, 0.99)
    lower = quantile(df.MomentValues, 0.01)
    df_winsor = df[findall(x -> lower < x && x < upper, df.MomentValues),:]
    return df_winsor.ParamValues, df_winsor.MomentValues
end
#---------------------------------------------------------------------------------------------------

#----- Moment Values For Parameter Marginals -----#
function MomentViolinPlots(midmicro::String, winsorize::Bool)
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv")) |> DataFrame
    sr = sr[findall(x -> x == midmicro, sr.Type),:]
    colors = ["blue", "red", "green", "magenta", "orange", "purple"]
    parameters = [("Nt", "Nᶜ"), ("Nv", "Nᶠ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    moments = [("Kurtosis", "Kurtosis"), ("KS", "KS"), ("Hurst", "Hurst Exponent"), ("GPH", "GPH Statistic"), ("ADF", "ADF Statistic"), ("GARCH", "GARCH"), ("Hill", "Hill Estimator"), ("Objective", "Objective Function")]
    for (i, (paramcol, paramlabel)) in enumerate(parameters)
        col = colors[i]
        for (momentcol, momentlabel) in moments
            params_sr = sr[:,paramcol]
            moments_sr = sr[:,momentcol]
            if winsorize
                params_sr, moments_sr = Winsorize(params_sr, moments_sr)
            end
            if paramcol == "Delta" || paramcol == "SigmaV"
                p = violin(string.(round.(params_sr, digits = 4)), moments_sr, quantiles = [0.025, 0.975], trim = true, show_median = true, tick_direction = :out, fillcolor = col, legend = false, xrotation = 30, yrotation = 30, tickfontsize = 12, guidefontsize = 22, xlabel = paramlabel, ylabel = momentlabel, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
                # xlabel!(paramlabel, fontsize = 22)
                # ylabel!(momentlabel, fontsize = 22)
                # boxplot!((round.(params_sr, digits = 4)), moments_sr, fillalpha = 0, marker = (1, :black, stroke(:black)), linewidth = 0, linecolor = :black, legend = false, group = params_sr)
            else
                p = violin(round.(params_sr, digits = 4), moments_sr, quantiles = [0.025, 0.975], trim = true, show_median = true, tick_direction = :out, fillcolor = col, legend = false, xrotation = 30, yrotation = 30, tickfontsize = 12, guidefontsize = 22, xlabel = paramlabel, ylabel = momentlabel, left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
                # xlabel!(paramlabel, fontsize = 22)
                # ylabel!(momentlabel, fontsize = 22)
                # boxplot!(round.(params_sr, digits = 4), moments_sr, fillalpha = 0, marker = (1, :black, stroke(:black)), linewidth = 0, linecolor = :black, legend = false)
            end
            savefig(p, "../Images/SensitivityAnalysis/Violin/NoKurtosis/" * midmicro * "Images/" * paramcol * momentcol * ".pdf")
        end
    end
end

# MomentViolinPlots("MicroPrice", true)
# MomentViolinPlots("MidPrice", true)
#---------------------------------------------------------------------------------------------------

#----- Moment Surfaces For Parameter Interactions -----#
function MomentInteractionSurfaces(midmicro::String, winsorize::Bool)
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv")) |> DataFrame
    sr = sr[findall(x -> x == midmicro, sr.Type),:]
    colors = ["blue", "red", "green", "magenta", "orange", "purple"]
    parameters = [("Nt", "Nᶜ"), ("Nv", "Nᶠ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    pairwise_combinations = collect(combinations(parameters, 2))
    moments = [("Kurtosis", "Kurtosis"), ("KS", "Kolmogorov-Smirnov"), ("Hurst", "Hurst Exponent"), ("GPH", "GPH Statistic"), ("ADF", "ADF Statistic"), ("GARCH", "GARCH"), ("Hill", "Hill Estimator")]
    for params in pairwise_combinations
        for (momentcol, momentlabel) in moments
            sr_grouped = groupby(sr, [params[1][1], params[2][1]]) |> gdf -> combine(gdf, Symbol(momentcol) => mean)
            surface = plot(unique(sr_grouped[:, params[1][1]]), unique(sr_grouped[:, params[2][1]]), reshape(sr_grouped[:, momentcol * "_mean"], (4,4)), seriestype = :surface, xlabel = params[1][2], ylabel = params[2][2], zlabel = momentlabel, colorbar = false, camera=(45,60), seriesalpha = 0.8, left_margin = 5Plots.mm, right_margin = 15Plots.mm, colorscale = "Viridis") # cgrad(ColorScheme((colorant"green", colorant"red", length=10))), color = cgrad([:green, :springgreen4, :firebrick2, :red]),
            savefig(surface, "../Images/SensitivityAnalysis/MomentInteractionSurfaces/" * midmicro * "Images/" * params[1][1] * params[2][1] * momentcol * ".pdf")
        end
    end
end

# MomentInteractionSurfaces("MicroPrice", false)
# MomentInteractionSurfaces("MidPrice", false)
# ---------------------------------------------------------------------------------------------------

#----- Objective Function Surfaces For Parameter Interactions -----#
function ObjectiveInteractionSurfaces(midmicro::String, winsorize::Bool)
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv")) |> DataFrame
    sr = sr[findall(x -> x == midmicro, sr.Type),:]
    colors = ["blue", "red", "green", "magenta", "orange", "purple"]
    parameters = [("Nt", "Nᶜ"), ("Nv", "Nᶠ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    pairwise_combinations = collect(combinations(parameters, 2))
    for params in pairwise_combinations
        sr_grouped = groupby(sr, [params[1][1], params[2][1]]) |> gdf -> combine(gdf, :Objective => mean)
        surface = plot(unique(sr_grouped[:, params[1][1]]), unique(sr_grouped[:, params[2][1]]), reshape(sr_grouped[:, "Objective_mean"], (4,4)), seriestype = :surface, xlabel = params[1][2], ylabel = params[2][2], zlabel = "Objective", colorbar = false, camera=(45,60), seriesalpha = 0.8, left_margin = 5Plots.mm, right_margin = 15Plots.mm, colorscale = "Viridis") # cgrad(ColorScheme((colorant"green", colorant"red", length=10))), color = cgrad([:green, :springgreen4, :firebrick2, :red]),
        savefig(surface, "../Images/SensitivityAnalysis/ObjectiveInteractionSurfaces/NoKurtosis/" * midmicro * "Images/" * params[1][1] * params[2][1] * "Objective.pdf")
    end
end

# ObjectiveInteractionSurfaces("MicroPrice", false)
# ObjectiveInteractionSurfaces("MidPrice", false)
#---------------------------------------------------------------------------------------------------

#----- Correlations Matrix -----#
function ParameterMomentCorrelationMatrix(midmicro::String, winsorize::Bool)
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv")) |> DataFrame
    sr = sr[findall(x -> x == midmicro, sr.Type),:]
    # variables = [("Nt", "Nᴸₜ"), ("Nv", "Nᴸᵥ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᵥ"), ("Mean", "Mean"), ("Std", "Std"), ("Kurtosis", "Kurtosis"), ("KS", "KS"), ("Hurst", "Hurst"), ("GPH", "GPH"), ("ADF", "ADF"), ("GARCH", "GARCH"), ("Hill", "Hill"), ("Objective", "Objective")]
    variables = [("Nt", "Nᶜ"), ("Nv", "Nᶠ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ"), ("Mean", "Mean"), ("Std", "Std"), ("KS", "KS"), ("Hurst", "Hurst"), ("GPH", "GPH"), ("ADF", "ADF"), ("GARCH", "GARCH"), ("Hill", "Hill"), ("Objective", "Objective")]
    sr = sr[:,first.(variables)]
    C = cor(Matrix(sr))
    (n,m) = size(C)
    H = heatmap(last.(variables), last.(variables), C, c = cgrad(:seismic, [0, 0.28, 0.56, 1]), xticks = (0.5:1:length(variables), last.(variables)), yticks = (0.5:1:length(variables), last.(variables)), xrotation = 45, yrotation = 0,  yflip=true, tickfontsize = 5, tick_direction = :out, alpha = 0.8)
    annotate!(H, [(j - 0.5, i - 0.5, text(round(C[i,j],digits=3), 5,:black, :center)) for i in 1:n for j in 1:m])
    savefig(H, "../Images/SensitivityAnalysis/CorrelationMatrix/CorrelationMatrix" * midmicro * ".pdf")
end 

# ParameterMomentCorrelationMatrix("MicroPrice", false)
# ParameterMomentCorrelationMatrix("MidPrice", false)
#---------------------------------------------------------------------------------------------------

#----- Exposure Matrix -----#
function ExposureMatrix(midmicro::String)
    sr = CSV.File(string("../Data/SensitivityAnalysis/SensitivityAnalysisResultsObj.csv")) |> DataFrame
    sr = sr[findall(x -> x == midmicro, sr.Type),:]
    moments = [("Mean", "Mean"), ("Std", "Std"), ("KS", "KS"), ("Hurst", "Hurst"), ("GPH", "GPH"), ("ADF", "ADF"), ("GARCH", "GARCH"), ("Hill", "Hill")]
    parameters = [("Nt", "Nᴸₜ"), ("Nv", "Nᴸᵥ"), ("Delta","δ"), ("Kappa", "κ"), ("Nu", "ν"), ("SigmaV", "σᶠ")]
    B = fill(0.0, length(parameters), length(moments))
    for (i, param) in enumerate(parameters)
        for (j, moment) in enumerate(moments)
            B[i,j] = cov(sr[:,first(param)], sr[:,first(moment)]) / var(sr[:,first(moment)])
        end
    end
    save("../Data/SensitivityAnalysis/B.jld", "B", B)
end

# ExposureMatrix("MicroPrice")
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------