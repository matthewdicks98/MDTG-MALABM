ENV["JULIA_COPY_STACKS"]=1
using DataFrames, CSV, Plots, Statistics

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/ReactiveABM.jl"); include(path_to_files * "Scripts/CoinTossXUtilities.jl")

#----- Simulate historical distributions for spread and volume -----# (ensure that CoinTossX has started)
function GenerateHistoricalDistributions(calibratedParameters::Parameters, simulations::Int64) 
    StartJVM()
    gateway = Login(1, 1)
    @time for i in 1:simulations
        try 
            # will use 5 seeds 
            seed = (i % 5) + 1
            println()
            println(string("Iteration = ", i))
            println(string("Seed = ", seed))
            println()
            sleep(1)
            @time simulate(parameters, gateway, false, false, true, seed = seed)
            GC.gc()             # perform garbage collection
        catch e
            Logout(gateway); StopCoinTossX()
            @error "Something went wrong" exception=(e, catch_backtrace())
            break
        end
    end
    Logout(gateway)
    StopCoinTossX()
end 
# set the parameters
Nᴸₜ = 8             
Nᴸᵥ = 6
Nᴴ = 30             
δ = 0.125           
κ = 3.389           
ν = 7.221               
m₀ = 10000          
σᵥ = 0.041         
λmin = 0.0005      
λmax = 0.05        
γ = Millisecond(1000) 
T = Millisecond(25000) 

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
# GenerateHistoricalDistributions(parameters, 365)
#---------------------------------------------------------------------------------------------------

#----- Historical Distributions -----# 
function HistoricalDistributions(numSpreadQuantiles::Int64, numVolumeQuantiles::Int64, plot::Bool, writeStates::Bool, id::Int64) # if doing multiple runs with different states this id identifies the run

    spreads_df = CSV.File(path_to_files * "/Data/RL/HistoricalDistributions/SpreadData.csv") |> DataFrame
    bid_volumes_df = CSV.File(path_to_files * "/Data/RL/HistoricalDistributions/BidVolumeData.csv") |> DataFrame
    ask_volumes_df = CSV.File(path_to_files * "/Data/RL/HistoricalDistributions/AskVolumeData.csv") |> DataFrame
    
    spreads = parse.(Float64, spreads_df[findall(x -> x != "Spread", spreads_df.Spread),:Spread])
    bid_volumes = parse.(Float64, bid_volumes_df[findall(x -> x != "BestBidVolume", bid_volumes_df.BestBidVolume),:BestBidVolume])
    ask_volumes = parse.(Float64, ask_volumes_df[findall(x -> x != "BestAskVolume", ask_volumes_df.BestAskVolume),:BestAskVolume])

    # spreads_sorted = sort(spreads)
    # println(length(spreads_sorted[1:findfirst(x -> x > 1.0, spreads_sorted)])/length(spreads_sorted))
    # println(length(spreads_sorted[1:findfirst(x -> x > 13.0, spreads_sorted)])/length(spreads_sorted))

    # bid_volumes_sorted = sort(bid_volumes)
    # println(length(bid_volumes_sorted[1:findfirst(x -> x > 6000, bid_volumes_sorted)])/length(bid_volumes_sorted))

    # ask_volumes_sorted = sort(ask_volumes)
    # println(length(ask_volumes_sorted[1:findfirst(x -> x > 6000, ask_volumes_sorted)])/length(ask_volumes_sorted))

    # create the quantiles
    split_interval_factor = numSpreadQuantiles # used to split the interval 0.6 < x <= 1 into enough quantiles to get required states
    spread_quantiles = nothing
    spread_quantiles_vals = nothing
    unique_spread_quantiles_vals = nothing
    while true
        spread_quantiles = [i for i in range(0.6,1,split_interval_factor)]        # 0 < x <= 0.6 is just made up of 1s
        spread_quantiles_vals = [quantile(spreads, q) for q in spread_quantiles]
        unique_spread_quantiles_vals = unique(spread_quantiles_vals)
        if length(unique_spread_quantiles_vals) == numSpreadQuantiles
            break
        else
            split_interval_factor += 1
        end
    end
    volume_quantiles = [i/numVolumeQuantiles for i in 1:numVolumeQuantiles]
    println("Spread Quantiles: " * join(string.(spread_quantiles), " ") * " | Number of quantiles = " * string(length(spread_quantiles)))
    println("Volume Quantiles: " * join(string.(volume_quantiles), " "))
    bid_quantiles_vals = [quantile(bid_volumes, q) for q in volume_quantiles]
    ask_quantiles_vals = [quantile(ask_volumes, q) for q in volume_quantiles]
    println("Spread Quantiles Values: " * join(string.(spread_quantiles_vals), " "))
    println("Unique Spread Quantile Values: " * join(string.(unique_spread_quantiles_vals), " "))
    println("Bid Quantiles Values: " * join(string.(bid_quantiles_vals), " "))
    println("Ask Quantiles Values: " * join(string.(ask_quantiles_vals), " "))

    # Find prob of being in each state
    spread_state_probs = round.(diff(pushfirst!([spread_quantiles[findlast(x -> x == i, spread_quantiles_vals)] for i in unique_spread_quantiles_vals], 0)), digits = 3)
    volume_state_probs = round.(diff(pushfirst!(volume_quantiles, 0)), digits = 3)
    println("Spread State Probabilities: " * join(string.(spread_state_probs), " ")) # shows probs less than quantile it is matched with and bigger than the one below (zero for the first)
    println("Volume State Probabilities: " * join(string.(volume_state_probs), " "))

    # plot the distributions
    if plot
        p1 = histogram(spreads, fillcolor = :green, linecolor = :green, xlabel = "Spread", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)
        vline!(spread_quantiles_vals)
        p2 = histogram(bid_volumes, fillcolor = :green, linecolor = :green, xlabel = "Best Bid Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)
        vline!(bid_quantiles_vals)
        p3 = histogram(ask_volumes, fillcolor = :green, linecolor = :green, xlabel = "Best Ask Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)
        vline!(ask_quantiles_vals)

        Plots.savefig(p1, path_to_files * "Images/RL/SpreadDist.pdf")
        Plots.savefig(p2, path_to_files * "Images/RL/BestBidVolume.pdf")
        Plots.savefig(p3, path_to_files * "Images/RL/BestAskVolume.pdf")
    end

    if writeStates 
        spread_states_df = DataFrame(SpreadStates = unique_spread_quantiles_vals, SpreadStateProbability = spread_state_probs)
        volume_states_df = DataFrame(BidVolumeStates = bid_quantiles_vals, AskVolumeStates = ask_quantiles_vals, VolumeStateProbability = volume_state_probs)
        CSV.write(path_to_files * "/Data/RL/SpreadVolumeStates/SpreadStates" * string(id) * ".csv", spread_states_df)
        CSV.write(path_to_files * "/Data/RL/SpreadVolumeStates/VolumeStates" * string(id) * ".csv", volume_states_df)
    end

end
HistoricalDistributions(5,5,false,true,1)
#---------------------------------------------------------------------------------------------------