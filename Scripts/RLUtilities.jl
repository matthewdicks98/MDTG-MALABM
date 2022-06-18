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
            @time simulate(calibratedParameters, gateway, false, false, true, seed = seed)
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
# Nᴸₜ = 8             
# Nᴸᵥ = 6
# Nᴴ = 30             
# δ = 0.125           
# κ = 3.389           
# ν = 7.221               
# m₀ = 10000          
# σᵥ = 0.041         
# λmin = 0.0005      
# λmax = 0.05        
# γ = Millisecond(1000) 
# T = Millisecond(25000) 

# parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
# GenerateHistoricalDistributions(parameters, 365)
#---------------------------------------------------------------------------------------------------

#----- Historical Distributions -----# 
function HistoricalDistributionsStates(numSpreadQuantiles::Int64, numVolumeQuantiles::Int64, print::Bool, plot::Bool, writeStates::Bool, id::Int64) # if doing multiple runs with different states this id identifies the run

    println("-------------------------------- Generating Spread and Volume States --------------------------------")

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
    bid_quantiles_vals = [quantile(bid_volumes, q) for q in volume_quantiles]
    ask_quantiles_vals = [quantile(ask_volumes, q) for q in volume_quantiles]

    # Find prob of being in each state
    spread_state_probs = round.(diff(pushfirst!([spread_quantiles[findlast(x -> x == i, spread_quantiles_vals)] for i in unique_spread_quantiles_vals], 0)), digits = 3)
    volume_state_probs = round.(diff(pushfirst!(volume_quantiles, 0)), digits = 3)
   
    if print
        println("Spread Quantiles: " * join(string.(spread_quantiles), " ") * " | Number of quantiles = " * string(length(spread_quantiles)))
        println("Volume Quantiles: " * join(string.(volume_quantiles), " "))
        println("Spread Quantiles Values: " * join(string.(spread_quantiles_vals), " "))
        println("Unique Spread Quantile Values: " * join(string.(unique_spread_quantiles_vals), " "))
        println("Bid Quantiles Values: " * join(string.(bid_quantiles_vals), " "))
        println("Ask Quantiles Values: " * join(string.(ask_quantiles_vals), " "))
        println("Spread State Probabilities: " * join(string.(spread_state_probs), " ")) # shows probs less than quantile it is matched with and bigger than the one below (zero for the first)
        println("Volume State Probabilities: " * join(string.(volume_state_probs), " "))
    end

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

    spread_states_df = DataFrame(SpreadStates = unique_spread_quantiles_vals, SpreadStateProbability = spread_state_probs)
    volume_states_df = DataFrame(BidVolumeStates = bid_quantiles_vals, AskVolumeStates = ask_quantiles_vals, VolumeStateProbability = volume_state_probs)

    # write states
    if writeStates 
        CSV.write(path_to_files * "/Data/RL/SpreadVolumeStates/SpreadStates" * string(id) * ".csv", spread_states_df)
        CSV.write(path_to_files * "/Data/RL/SpreadVolumeStates/VolumeStates" * string(id) * ".csv", volume_states_df)
    end

    println("----------------------------------------------------------------")

    return spread_states_df, volume_states_df

end
# spread_states_df, volume_states_df = HistoricalDistributionsStates(5,5,false,false,1)
#---------------------------------------------------------------------------------------------------

#----- Return current state -----# 
function GetSpreadState(LOB::LOBState, rlParameters::RLParameters)
    lower = 0
    state_counter = 1
    sₙ = 1
    for upper in rlParameters.spread_states_df.SpreadStates
        if lower < LOB.sₜ && LOB.sₜ <= upper # check 
            sₙ = state_counter
            break
        else
            state_counter += 1
        end
        lower = upper
    end
    if state_counter > rlParameters.B # if the spread is greater than the max spread of historical dist then assign it to the last state
        sₙ = rlParameters.B
    end
    if LOB.sₜ <= 0
        sₙ = 1
    end
    println("Spread = ", LOB.sₜ)
    println("Spread State = ", sₙ)
    return sₙ
end
function GetVolumeState(LOB::LOBState, rlParameters::RLParameters, rlAgent::RL)
    lower = 0
    v = 0 # volume of best bid or ask depending on the agents action type
    # if there is no volume on the other side then set Vₙ to state 1
    if rlAgent.actionType == "Sell" && length(LOB.bids) == 0
        v = 0
    elseif rlAgent.actionType == "Buy" && length(LOB.asks) == 0
        v = 0
    else
        rlAgent.actionType == "Sell" ?  v = sum(order.volume for order in values(LOB.bids) if order.price == LOB.bₜ) : v = sum(order.volume for order in values(LOB.asks) if order.price == LOB.aₜ)
    end
    column_name = "" # column name in the volume state data frame
    rlAgent.actionType == "Sell" ? column_name = "BidVolumeStates" : column_name = "AskVolumeStates"
    state_counter = 1
    vₙ = 1
    for upper in rlParameters.volume_states_df[:,column_name]
        if lower < v && v <= upper # check 
            vₙ = state_counter
            break
        else
            state_counter += 1
        end
        lower = upper
    end
    if state_counter > rlParameters.W # if the spread is greater than the max volume of historical dist then assign it to the last state
        vₙ = rlParameters.W
    end
    if v <= 0
        vₙ = 1
    end
    println("Volume = ", v)
    println("Volume State = ", vₙ)
    return vₙ
end
function GetTimeState(t::Int64, rlParameters::RLParameters)
    # get the total time to execute parameter T, divide it by number of time states to get the time state interval length
    τ = Int64(rlParameters.T.value/rlParameters.numT) # this will break if it is not divisible
    lower_t = 0
    upper_t = 0 + τ
    tₙ = 0
    state_counter = 1
    for i in 1:numT # numT
        if lower_t < t && t <= upper_t
            tₙ = state_counter
            break
        else
            state_counter += 1
        end
        lower_t = upper_t
        upper_t += τ
    end    
    if state_counter > rlParameters.numT # if the spread is greater than the max volume of historical dist then assign it to the last state
        tₙ = rlParameters.numT
    end
    if t <= 0
        tₙ = 1
    end
    println("Time = ", t)
    println("Time State = ", tₙ)
    return tₙ
end
function GetInventoryState(i::Int64, rlParameters::RLParameters)
    interval = Int64(rlParameters.V/rlParameters.I) # this will break if it is not divisible
    lower_i = 0
    upper_i = 0 + interval
    iₙ = 0
    state_counter = 1
    for j in 1:I # numT
        if lower_i < i && i <= upper_i
            iₙ = state_counter
            break
        else
            state_counter += 1
        end
        lower_i = upper_i
        upper_i += interval
    end    
    if state_counter > rlParameters.I # if the spread is greater than the max volume of historical dist then assign it to the last state
        iₙ = rlParameters.I
    end
    if i <= 0
        iₙ = 1
    end
    println("Inventory = ", i)
    println("Inventory State = ", iₙ)
    return iₙ
end
function GetState(LOB::LOBState, t::Int64, i::Int64, rlParameters::RLParameters, rlAgent::RL) # t and i are the remaining time and inventory
    
    # find the spread state
    sₙ = GetSpreadState(LOB, rlParameters)

    # find the volume state
    vₙ = GetVolumeState(LOB, rlParameters, rlAgent)

    # find the time state
    tₙ = GetTimeState(t, rlParameters)

    # find the inventory remaining state
    iₙ = GetInventoryState(i, rlParameters)

    # return the state vector <tₙ, iₙ, sₙ, vₙ>
    println([tₙ, iₙ, sₙ, vₙ])
    return [tₙ, iₙ, sₙ, vₙ]

end
#---------------------------------------------------------------------------------------------------

#----- Epsilon Greedy Policy -----# 

#---------------------------------------------------------------------------------------------------

#----- Simulate ABM with RL agent -----# 

#---------------------------------------------------------------------------------------------------

# set the parameters
Nᴸₜ = 8             # [3,6,9,12]
Nᴸᵥ = 6
Nᴴ = 30             # fixed at 30
δ = 0.125           # 0.01, 0.07, 0.14, 0.2
κ = 3.389           # 2, 3, 4, 5
ν = 7.221               # 2, 4, 6, 8
m₀ = 10000          # fixed at 10000
σᵥ = 0.041         # 0.0025, 0.01, 0.0175, 0.025
λmin = 0.0005       # fixed at 0.0005
λmax = 0.05         # fixed at 0.05
γ = Millisecond(1000) # fixed at 1000
T = Millisecond(25000) # fixed at 25000 
seed = 1 # 125 has price decrease

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

# Rl parameters
Nᵣₗ = 1                      # num rl agents
startTime = Millisecond(10)  # start time for RL agents
T = Millisecond(100)         # execution time for RL agents 
numT = 10                    # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown)
V = 1000                     # volume to trade in each execution
I = 10                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown)
B = 5                        # number of spread states
W = 5                        # number of volume states
A = 10                       # number of action states

spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,false,1)
type = "Sell"

rlParameters = RLParameters(Nᵣₗ, startTime, T, numT, V, I, B, W, A, spread_states_df, volume_states_df, type)

# set the parameters that dictate output
print_and_plot = false                    # Print out useful info about sim and plot simulation time series info
write_messages = false                             # Says whether or not the messages data must be written to a file
write_volume_spread = false
rlTraders = true

# run the simulation
StartJVM()
gateway = Login(1,1)
try 
    @time simulate(parameters, rlParameters, gateway, rlTraders, print_and_plot, write_messages, write_volume_spread, seed = seed)
catch e
    @error "Something went wrong" exception=(e, catch_backtrace())
finally
    Logout(gateway)
    # StopCoinTossX()
end