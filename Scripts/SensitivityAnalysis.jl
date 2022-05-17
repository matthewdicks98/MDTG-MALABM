ENV["JULIA_COPY_STACKS"]=1
using ProgressMeter, CSV, Plots, DataFrames, StatsPlots, Statistics, ColorSchemes, Dates
using LinearAlgebra: diag, inv

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

#----- Sensitivity analysis -----#
function SensitivityAnalysis(empericalLogReturns::DataFrame, empericalMoments::Dict, parameterCombinations::Vector{Parameters}, parameterCombinationsRange::Vector{Int64})
    # StartCoinTossX(false, false)
    StartJVM()
    gateway = Login(1, 1)
    open("../Data/SensitivityAnalysis/SensitivityAnalysisResults.csv", "w") do file
        println(file, "Type,Nt,Nv,Nh,Delta,Kappa,Nu,M0,SigmaV,LambdaMin,LambdaMax,Gamma,T,Seed,Mean,Std,Kurtosis,KS,Hurst,GPH,ADF,GARCH,Hill")
        for (i, parameters) in enumerate(parameterCombinations[1:5]) # [parameterCombinationsRange[1]:parameterCombinationsRange[2]])
            try 
                seed = 1
                @time midPrices, microPrices = simulate(parameters, gateway, false, false, seed = seed)
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
parameterCombinationsRange = map(x -> parse(Int64, x), ARGS)

# make sure these are the same for the stylized facts and Calibration
date = DateTime("2019-07-08")
startTime = date + Hour(9) + Minute(1)
endTime = date + Hour(16) + Minute(50) # Hour(17) ###### Change to 16:50

# empericalLogReturns, empericalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)

NᴸₜRange = [3,6,9,12]
NᴸᵥRange = [3,6,9,12]
δRange = collect(range(0.01, 0.2, length = 4))
κRange = collect(range(2, 5, length = 4))
νRange = collect(range(2, 8, length = 4))
σᵥRange = collect(range(0.0025, 0.025, length = 4))

parameterCombinations = GenerateParameterCombinations(NᴸₜRange, NᴸᵥRange, δRange, κRange, νRange, σᵥRange)

for p in parameterCombinations[2499:2501]
    println(p)
end

# @time SensitivityAnalysis(empericalLogReturns, empericalMoments, parameterCombinations, parameterCombinationsRange)