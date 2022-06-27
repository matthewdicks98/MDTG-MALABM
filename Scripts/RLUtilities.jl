ENV["JULIA_COPY_STACKS"]=1
using DataFrames, CSV, Plots, Statistics, DataStructures, JLD

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

    return spread_states_df, volume_states_df

end
# spread_states_df, volume_states_df = HistoricalDistributionsStates(5,5,false,false,1)
#---------------------------------------------------------------------------------------------------

#----- Return actions -----# 
function GenerateActions(A::Int64)
    println("-------------------------------- Generating Actions --------------------------------")
    actions = Dict{Int64, Float64}()
    for (i, p) in enumerate(range(0, 1, A))
        actions[i] = p^2
    end
    return actions
end
#---------------------------------------------------------------------------------------------------

#----- Given the last RL messages return the volume traded and the profit/cost of the trade -----# 
function ProcessMessages(messages::Vector{String}, rlAgent::RL)

    total_volume_traded = 0
    sum_price_volume = 0
    trade_message = ""

    for message in messages

        # take the time out of the message
        msg = split(message, "|")[2:end]

        #msg = split(message, "|")
        fields = split(msg[1], ",")

        # need 2 types (Type is the original one and type is the one that changes (crossing LOs))
        trader = fields[3]

        # check if it is an order for the current RL trader
        if trader == rlAgent.traderMnemonic

            # get the executions
            executions = msg[2:end]
            if executions == [""]
                return 0, 0, ""      
            end

            for execution in executions
                executionFields = split(execution, ",")
                id = parse(Int, executionFields[1]); price = parse(Int, executionFields[2]); volume = parse(Int, executionFields[3])
                total_volume_traded += volume
                sum_price_volume += price * volume
            end

            trade_message = message

            break # only 1 order (message) per event loop per trader

        end

    end

    if total_volume_traded == 0 && sum_price_volume == 0
        return 0, 0, ""
    else
        return total_volume_traded, sum_price_volume, trade_message
    end
end
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
    # println("Spread = ", LOB.sₜ)
    # println("Spread State = ", sₙ)
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
    if state_counter > rlParameters.W # if the volume is greater than the max volume of historical dist then assign it to the last state
        vₙ = rlParameters.W
    end
    if v <= 0
        vₙ = 1
    end
    # println("Volume = ", v, " | At = ", LOB.bₜ)
    # println("Volume State = ", vₙ)
    return vₙ
end
function GetTimeState(t::Int64, rlParameters::RLParameters)
    # get the total time to execute parameter T, divide it by number of time states to get the time state interval length
    τ = Int64(rlParameters.T.value/rlParameters.numT) # this will break if it is not divisible
    lower_t = 0
    upper_t = 0 + τ
    tₙ = 0
    state_counter = 1
    for i in 1:rlParameters.numT # numT
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
        tₙ = 0  # this means that we will take an action such that we transition us to the terminal state with probability 1
    end
    # println("Time = ", t)
    # println("Time State = ", tₙ)
    return tₙ
end
function GetInventoryState(i::Int64, rlParameters::RLParameters)
    interval = Int64(rlParameters.V/rlParameters.I) # this will break if it is not divisible
    lower_i = 0
    upper_i = 0 + interval
    iₙ = 0
    state_counter = 1
    for j in 1:rlParameters.I # numT
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
        iₙ = 0 # this will then mean that we are going to return the terminal state, we have stopped trading
    end
    # println("Inventory = ", i)
    # println("Inventory State = ", iₙ)
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

    # check if terminal and return the state vector <tₙ, iₙ, sₙ, vₙ>
    if iₙ == 0
        return [0, 0, 0, 0], true # return terminal state and say that we are done
    else
        return [tₙ, iₙ, sₙ, vₙ], false # return state and say that we still have inventory to trade
    end

end
#---------------------------------------------------------------------------------------------------

#----- Epsilon Greedy Policy -----# 
function EpisilonGreedyPolicy(Q::DefaultDict, state::Vector{Int64}, epsilon::Float64, rlAgent::RL)
    # create and epsilon greedy policy for a given state
    num_actions_state = length(Q[state])
    policy = fill(epsilon / num_actions_state, num_actions_state) # each state has an equal prob of being chosen
    # get the action with the lowest (highest) value for selling (buying)
    a_star = nothing
    if rlAgent.actionType == "Sell" # want to maximize the profit
        a_star = argmax(Q[state]) # change so that action 1 isnt always selected for a new state
    elseif rlAgent.actionType == "Buy" # want to mimimize the cost (might need to change)
        a_star = argmin(Q[state])
    end
    # the optimal action will be chosen with this probability
    policy[a_star] = 1 - epsilon + (epsilon / num_actions_state)
    return policy
end

#---------------------------------------------------------------------------------------------------

#----- Test ABM Simulation with RL agent -----# 
function TestRunRLABM()
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
    seed = 8 # 6, 8, 9

    parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

    # Rl parameters
    Nᵣₗ = 1                      # num rl agents
    startTime = Millisecond(0)   # start time for RL agents (keep it at the start of the sim until it is needed to have multiple)
    rlT = Millisecond(24500)        # execution time for RL agents (needs to ensure that RL agents finish before other agents to allow for correct computation of final cost)
    numT = 10                   # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
    V = 50000                     # volume to trade in each execution
    I = 10                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
    B = 5                        # number of spread states
    W = 5                        # number of volume states
    A = 10                       # number of action states

    spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,false,1)
    actions = GenerateActions(A)
    actionType = "Sell"
    ϵ = 0.1            # used in epsilon greedy algorithm
    discount_factor = 1 # used in Q update (discounts future rewards)
    α = 0.5             # used in Q update

    rlParameters = RLParameters(Nᵣₗ, startTime, rlT, numT, V, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, ϵ, discount_factor, α)

    # set the parameters that dictate output
    print_and_plot = true                    # Print out useful info about sim and plot simulation time series info
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
end
# TestRunRLABM()
#---------------------------------------------------------------------------------------------------

#----- Train RL Agent in an ABM -----# 
function TrainRL(parameters::Parameters, rlParameters::RLParameters, numEpisodes::Int64)

    println("-------------------------------- Training RL Agent --------------------------------")

    # define storage
    rl_results = Vector{Dict}()

    prev_q = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A))

    # run the simulation
    StartJVM()
    gateway = Login(1,1)
    try 
        for i in 1:numEpisodes

            println()
            println("------------------------ ", i, " ------------------------")
            println()

            @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, true, false, false, false, seed = 1)           
            
            # print some test stuff
            # println("Initial Q ")
            # for key in keys(rlParameters.initialQ)
            #     println(key, " => ",rlParameters.initialQ[key], " | Total actions = ", sum(rlParameters.initialQ[key]))
            # end
            println()
            println("Resulting Q")
            for key in keys(rl_result["Q"])
                println(key, " => ",rl_result["Q"][key], " | Total actions = ", sum(rl_result["Q"][key]))
            end
            println()
            # if i > 1
            #     println(prev_q == rlParameters.initialQ)
                
            #     if prev_q == rlParameters.initialQ
            #         println()
            #         println()
            #         println("Prev Q")
            #         for key in keys(prev_q)
            #             println(key, " => ",prev_q[key], " | Total actions = ", sum(prev_q[key]))
            #         end
            #         println()
            #         println("Initial Q")
            #         for key in keys(rlParameters.initialQ)
            #             println(key, " => ",rlParameters.initialQ[key], " | Total actions = ", sum(rlParameters.initialQ[key]))
            #         end
            #     else
            #         println("Q Error")
            #         return
            #     end
            # end

            # used to test the passing of Q from one simulation to another
            for key in keys(rl_result["Q"])
                prev_q[key] = copy(rl_result["Q"][key])
            end

            # for the next simulation use the learnt Q from the previous as the new Q (make a copy, copy is made in the simulation)
            rlParameters.initialQ = rl_result["Q"]

            # write the RL data from that iteration to a jld database
            @time begin
                if isfile(path_to_files * "/Data/RL/Training/Results.jld")
                    f = jldopen(path_to_files * "/Data/RL/Training/Results.jld", "r+")
                    rl_result["Q"] = Dict(rl_result["Q"])
                    f[string(i)] = rl_result
                    close(f)
                else
                    rl_result["Q"] = Dict(rl_result["Q"])
                    save(path_to_files * "/Data/RL/Training/Results.jld", string(i), rl_result)
                end
            end
            # garbage collect
            GC.gc()
        end
    catch e
        @error "Something went wrong" exception=(e, catch_backtrace())
    finally
        Logout(gateway)
        # StopCoinTossX()
    end 

end

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
seed = 8 # 6, 8, 9

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

# Rl parameters
Nᵣₗ = 1                      # num rl agents
startTime = Millisecond(0)   # start time for RL agents (keep it at the start of the sim until it is needed to have multiple)
rlT = Millisecond(24500)     # 24500 execution duration for RL agents (needs to ensure that RL agents finish before other agents to allow for correct computation of final cost)
numT = 10                   # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
V = 50000                     # volume to trade in each execution
I = 10                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
B = 5                        # number of spread states
W = 5                        # number of volume states
A = 15                       # number of action states

spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,false,1)
actions = GenerateActions(A)
actionType = "Sell"
ϵ = 1            # used in epsilon greedy algorithm
discount_factor = 1 # used in Q update (discounts future rewards)
α = 0.5             # used in Q update
initialQ = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, A))

rlParameters = RLParameters(Nᵣₗ, initialQ, startTime, rlT, numT, V, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, ϵ, discount_factor, α)

# set the parameters that dictate output
print_and_plot = true                    # Print out useful info about sim and plot simulation time series info
write_messages = false                             # Says whether or not the messages data must be written to a file
write_volume_spread = false
rlTraders = true

# rl training parameters
numEpisodes = 5

# train the agent
@time TrainRL(parameters, rlParameters, numEpisodes)

#---------------------------------------------------------------------------------------------------

function RLTestResults()
    l = load(path_to_files * "Data/RL/Training/Results.jld")
    n = length(l)
    println("---------------- Number of Actions ------------")
    for i in 1:n
        println(string(i), " => ", l[string(i)]["NumberActions"])
    end
    println()
    println("---------------- Number of Trades ------------")
    for i in 1:n
        println(string(i), " => ", l[string(i)]["NumberTrades"])
    end
    println()
    println("---------------- Actions ------------")
    for i in 1:n
        println(string(i), " => ", l[string(i)]["Actions"])
    end
    println()
    println("---------------- Total Reward ------------")
    for i in 1:n
        println(string(i), " => ", l[string(i)]["TotalReward"])
    end
    println()
    println("---------------- Rewards ------------")
    for i in 1:n
        println(string(i), " => ", l[string(i)]["Rewards"])
    end
    println()
    println("---------------- Number of States in Q ------------")
    for i in 1:n
        println(string(i), " => ", length(l[string(i)]["Q"]))
    end
    println()
    println("---------------- First Q ------------")
    for key in keys(l[string(1)]["Q"])
        println(key, " => ", l[string(1)]["Q"][key])
    end
    println()
    println("---------------- Last Q ------------")
    for key in keys(l[string(n)]["Q"])
        println(key, " => ", l[string(n)]["Q"][key])
    end

end
# RLTestResults()
