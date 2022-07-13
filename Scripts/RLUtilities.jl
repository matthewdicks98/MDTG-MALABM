ENV["JULIA_COPY_STACKS"]=1
using DataFrames, CSV, Plots, Statistics, DataStructures, JLD, Plots.PlotMeasures

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
function GenerateActions(A::Int64, maxVolunmeIncrease::Float64)
    println("-------------------------------- Generating Actions --------------------------------")
    actions = Dict{Int64, Float64}()
    for (i, p) in enumerate(range(0, maxVolunmeIncrease, A))
        actions[i] = p
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
    # place the reamining time into a state
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
function TrainRL(parameters::Parameters, rlParameters::RLParameters, numEpisodes::Int64, rlTraders::Bool, iterationsPerWrite::Int64, steps::Vector{Int64}, stepSizes::Vector{Float64})

    println("-------------------------------- Training RL Agent --------------------------------")

    rl_training = true

    # set storage for single agent rl_results (change when multiple)
    rl_results = Dict{Int64,Dict}()

    # store the prev q for checking
    prev_q = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A))

    # set the epsilon and the steps counter
    ϵ₀ = rlParameters.ϵ
    steps_counter = 1

    # run the simulation
    StartJVM()
    gateway = Login(1,1)
    try 
        for i in 1:numEpisodes

            println()
            println("------------------------ ", i, " ----- ", rlParameters.ϵ, " ------------------------")
            println(run(`free -m`))
            println()

            # set a variable to store the rl result
            rl_result = Dict()

            # set the seed for the episode (represents changes in the ABM agents views) (use 10% of the number of episodes as distinct seeds)
            seed = Int64((i % 10 + 1))

            # determine if you need to write the messages (always write first and last)
            if i % iterationsPerWrite == 0 || i == numEpisodes || i == 1
                @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, rlTraders, rl_training, false, true, false, seed = seed, iteration = i)
            else
                @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, rlTraders, rl_training, false, false, false, seed = seed, iteration = i)
            end           

            # update ϵ
            rlParameters.ϵ = max(rlParameters.ϵ - (ϵ₀ - (1 - stepSizes[steps_counter]) * ϵ₀)/steps[steps_counter], 0)

            if i == sum(steps[1:steps_counter]) # move to the next step regime
                steps_counter += 1
            end

            # println()
            # println("Resulting Q")
            # for key in keys(rl_result["Q"])
            #     println(key, " => ",rl_result["Q"][key], " | Total actions = ", sum(rl_result["Q"][key]))
            # end
            # println()
            if i > 1                
                if prev_q != rlParameters.initialQ
                    println("Q Error")
                    return
                end
            end

            # used to test the passing of Q from one simulation to another
            for key in keys(rl_result["Q"])
                prev_q[key] = copy(rl_result["Q"][key])
            end

            # for the next simulation use the learnt Q from the previous as the new Q (make a copy, copy is made in the simulation)
            rlParameters.initialQ = rl_result["Q"]

            # save the rl_result to the vector
            rl_result["Q"] = Dict(rl_result["Q"])
            push!(rl_results, i => rl_result)

            # write the RL data from that iteration to a jld database
            # if isfile(path_to_files * "/Data/RL/Training/Results.jld")
            #     @time f = jldopen(path_to_files * "/Data/RL/Training/Results.jld", "r+")
            #     rl_result["Q"] = Dict(rl_result["Q"])
            #     @time f[string(i)] = rl_result
            #     close(f)
            # else
            #     rl_result["Q"] = Dict(rl_result["Q"])
            #     save(path_to_files * "/Data/RL/Training/Results.jld", string(i), rl_result)
            # end

            # garbage collect
            GC.gc()
        end
    catch e
        @error "Something went wrong" exception=(e, catch_backtrace())
    finally
        Logout(gateway)
        StopCoinTossX()
        # write results to a file
        @time save(path_to_files * "/Data/RL/Training/Results.jld", "rl_results", rl_results)
    end 

end

# # set the parameters
# Nᴸₜ = 8             # [3,6,9,12]
# Nᴸᵥ = 6
# Nᴴ = 30             # fixed at 30
# δ = 0.125           # 0.01, 0.07, 0.14, 0.2
# κ = 3.389           # 2, 3, 4, 5
# ν = 7.221               # 2, 4, 6, 8
# m₀ = 10000          # fixed at 10000
# σᵥ = 0.041         # 0.0025, 0.01, 0.0175, 0.025
# λmin = 0.0005       # fixed at 0.0005
# λmax = 0.05         # fixed at 0.05
# γ = Millisecond(1000) # fixed at 25000
# T = Millisecond(25000) # fixed at 25000 
# seed = 1 # 6, 8, 9

# parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

# # Rl parameters
# Nᵣₗ = 1                      # num rl agents
# startTime = Millisecond(0)   # start time for RL agents (keep it at the start of the sim until it is needed to have multiple)
# rlT = Millisecond(24500)     # 24500 execution duration for RL agents (needs to ensure that RL agents finish before other agents to allow for correct computation of final cost)
# numT = 10                    # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
# V = 59850                    # (266/2 * 450) volume to trade in each execution (ensure it is large enough so that price impact occurs at higher TWAP volumes and lower TWAP volumes no price impact)
# I = 10                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
# B = 5                        # number of spread states
# W = 5                        # number of volume states
# A = 9                       # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)

# spread_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/SpreadStates1.csv") |> DataFrame
# volume_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/VolumeStates1.csv") |> DataFrame
# # spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,true,1)
# actions = GenerateActions(A, maxVolunmeIncrease)
# actionType = "Sell"
# ϵ₀ = 1            # used in epsilon greedy algorithm
# discount_factor = 1 # used in Q update (discounts future rewards)
# α = 0.1             # used in Q update (α = 0.1, 0.01, 0.5)
# initialQ = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, A))
# numDecisions = 450 # each agent has approx 450 decisions to make per simulation, Ntwap = V / numDecisions (this is fixed, but need to get estimated for new hardware)
# Ntwap = V / numDecisions

# rlParameters = RLParameters(Nᵣₗ, initialQ, startTime, rlT, numT, V, Ntwap, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, ϵ₀, discount_factor, α)

# # rl training parameters
# numEpisodes = 2
# steps = [75, 175, 50, 100]    # number of steps for each percentage decrease
# stepSizes = [0.1, 0.8, 0.09, 0]   # Percentage decrease over the number of steps
# iterationsPerWrite = 100

# # set the parameters that dictate output
# print_and_plot = false                    # Print out useful info about sim and plot simulation time series info
# write_messages = false                             # Says whether or not the messages data must be written to a file
# write_volume_spread = false
# rlTraders = true

# # train the agent
# @time TrainRL(parameters, rlParameters, numEpisodes, rlTraders, iterationsPerWrite, steps, stepSizes)

#---------------------------------------------------------------------------------------------------
 
#---------------------------------------------------------------------- Visualizations ----------------------------------------------------------------------#

#----- Plot the RL training results -----# 
function PlotRLConvergenceResults(actionsMap::Dict)

    # read in data into a dict
    data = Dict()
    Vs = [200, 100, 50]
    num_states = [5,10]
    for V in Vs
        for num_state in num_states
            @time d = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state) * ".jld")["rl_results"]
            push!(data, "Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state) => d)
        end
    end

    # convergence of rewards
    reward_plots = []
    Vs = [200, 100, 50]
    for V in Vs
        rewards5 = Vector{Float64}()
        rewards10 = Vector{Float64}()
        # @time l5 = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S5.jld")["rl_results"]
        # @time l10 = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S10.jld")["rl_results"]
        l5 = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S5"]
        l10 = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S10"]
        n = length(l10)
        for i in 1:n
            push!(rewards5, l5[i]["TotalReward"])
            push!(rewards10, l10[i]["TotalReward"])
        end
        if V == 200 # used to adjust the legend
            p = plot(rewards5, fillcolor = :red, linecolor = :red, legend = :topright, xlabel = "Episodes", ylabel = "Reward", title = "Volume = " * string(V * 450), titlefontsize = 6, label = "T,I,B,W = 5", legendfontsize = 4, fg_legend = :transparent)
            plot!(rewards10, fillcolor = :blue, linecolor = :blue, legend = :topright, xlabel = "Episodes", ylabel = "Reward", title = "Volume = " * string(V * 450), titlefontsize = 6, label = "T,I,B,W = 10", legendfontsize = 4, fg_legend = :transparent, )
            hline!([V * 450 * 10000], linecolor = :black, label = "IS", linestyle = :dash)
        else
            p = plot(rewards5, fillcolor = :red, linecolor = :red, legend = :bottomright, xlabel = "Episodes", ylabel = "Reward", title = "Volume = " * string(V * 450), titlefontsize = 6, label = "T,I,B,W = 5", legendfontsize = 4, fg_legend = :transparent)
            plot!(rewards10, fillcolor = :blue, linecolor = :blue, legend = :bottomright, xlabel = "Episodes", ylabel = "Reward", title = "Volume = " * string(V * 450), titlefontsize = 6, label = "T,I,B,W = 10", legendfontsize = 4, fg_legend = :transparent, )
            hline!([V * 450 * 10000], linecolor = :black, label = "IS", linestyle = :dash)
        end
        
        push!(reward_plots, p)
    end

    reward_plot = plot(reward_plots..., layout = grid(3,1), guidefontsize = 5, tickfontsize = 5)
    savefig(reward_plot, path_to_files * "/Images/RL/RewardConvergence.pdf")

    # plot the convergence in the number of states and trades
    num_states = [5,10]
    Vs = [200, 100, 50]
    num_states_dict = Dict()
    num_trades_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            num_states = Vector{Float64}()
            num_trades = Vector{Float64}()
            for i in 1:n
                push!(num_states, length(l[i]["Q"]))
                push!(num_trades, l[i]["NumberTrades"])
            end
            push!(num_states_dict, string(num_state) * "_" * string(V) => num_states)
            push!(num_trades_dict, string(num_state) * "_" * string(V) => num_trades)
        end
    end
    #plot the number of states convergence
    num_states_plots = plot(1:length(num_states_dict["5_200"]), num_states_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = "T,I,B,W = 5, Volume = 200", legend = :bottomleft, fg_legend = :transparent, xlabel = "Episodes", ylabel = "# States (T,I,B,W = 5)", title = "# States per Episode", right_margin = 12mm)
    plot!(1:length(num_states_dict["5_100"]), num_states_dict["5_100"], fillcolor = :red, linecolor = :red, label = "T,I,B,W = 5, Volume = 100", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_states_dict["5_50"]), num_states_dict["5_50"], fillcolor = :green, linecolor = :green, label = "T,I,B,W = 5, Volume = 50", legend = :bottomleft, fg_legend = :transparent)
    subplot = twinx()
    plot!(subplot, 1:length(num_states_dict["10_200"]), num_states_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = "T,I,B,W = 10, Volume = 200", ylabel = "# States (T,I,B,W = 10)", legend = :bottomright, fg_legend = :transparent)
    plot!(subplot, 1:length(num_states_dict["10_100"]), num_states_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = "T,I,B,W = 10, Volume = 100", legend = :bottomright, fg_legend = :transparent)
    plot!(subplot, 1:length(num_states_dict["10_50"]), num_states_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = "T,I,B,W = 10, Volume = 50", legend = :bottomright, fg_legend = :transparent)
    savefig(num_states_plots, path_to_files * "/Images/RL/NumberStatesConvergence.pdf")

    # plot the number of trades convergence
    num_states_plots = plot(1:length(num_trades_dict["10_200"]), num_trades_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = "T,I,B,W = 10, Volume = 200", legend = :bottomright, fg_legend = :transparent, xlabel = "Episodes", ylabel = "# Trades", title = "# Trades per Episode")
    plot!(1:length(num_trades_dict["10_100"]), num_trades_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = "T,I,B,W = 10, Volume = 100", legend = :bottomright, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["10_50"]), num_trades_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = "T,I,B,W = 10, Volume = 50", legend = :bottomright, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_200"]), num_trades_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = "T,I,B,W = 5, Volume = 200", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_100"]), num_trades_dict["5_100"], fillcolor = :red, linecolor = :red, label = "T,I,B,W = 5, Volume = 100", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_50"]), num_trades_dict["5_50"], fillcolor = :green, linecolor = :green, label = "T,I,B,W = 5, Volume = 50", legend = :bottomleft, fg_legend = :transparent)
    savefig(num_states_plots, path_to_files * "/Images/RL/NumberTradesConvergence.pdf")

    # convergence of policy (difference between best action in each state in consecutive iterations)
    num_states = [5,10]
    Vs = [200, 100, 50]
    policy_diffs_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            p_diffs = Vector{Float64}()
            for i in 2:n
                prev_q = l[i-1]["Q"]
                prev_q_state_values = getindex.(Ref(l[i-1]["Q"]), keys(prev_q))
                curr_q_state_values = getindex.(Ref(l[i]["Q"]), keys(prev_q))
                prev_policy = argmax.(prev_q_state_values)
                current_policy = argmax.(curr_q_state_values)
                p_diff = sum(prev_policy .!= current_policy) / length(prev_policy) 
                push!(p_diffs, p_diff)
            end
            push!(policy_diffs_dict, string(num_state) * "_" * string(V) => p_diffs)
        end
    end
    policy_diffs_plots = plot(1:length(policy_diffs_dict["5_200"]), policy_diffs_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = "T,I,B,W = 5, Volume = 200", fg_legend = :transparent, xlabel = "Episodes", ylabel = "Policy Differences", title = "1 Step Policy Differences")
    plot!(1:length(policy_diffs_dict["5_100"]), policy_diffs_dict["5_100"], fillcolor = :red, linecolor = :red, label = "T,I,B,W = 5, Volume = 100")
    plot!(1:length(policy_diffs_dict["5_50"]), policy_diffs_dict["5_50"], fillcolor = :green, linecolor = :green, label = "T,I,B,W = 5, Volume = 50")
    plot!(1:length(policy_diffs_dict["10_200"]), policy_diffs_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = "T,I,B,W = 10, Volume = 200")
    plot!(1:length(policy_diffs_dict["10_100"]), policy_diffs_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = "T,I,B,W = 10, Volume = 100")
    plot!(1:length(policy_diffs_dict["10_50"]), policy_diffs_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = "T,I,B,W = 10, Volume = 50")
    savefig(policy_diffs_plots, path_to_files * "/Images/RL/PolicyConvergence.pdf")

    # convergence of Q (difference between best action in each state in consecutive iterations)
    num_states = [5,10]
    Vs = [200, 100, 50]
    q_diffs_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            # convergence of q
            q_diffs = Vector{Float64}()
            for i in 2:n
                prev_q = l[i-1]["Q"]
                prev_q_state_values = getindex.(Ref(l[i-1]["Q"]), keys(prev_q))
                curr_q_state_values = getindex.(Ref(l[i]["Q"]), keys(prev_q))
                q_diff = [sum(abs.(s_diff)) / length(prev_q_state_values[1]) for s_diff in (curr_q_state_values .- prev_q_state_values)]
                push!(q_diffs, sum(q_diff) / length(prev_q))
            end
            push!(q_diffs_dict, string(num_state) * "_" * string(V) => q_diffs)
        end
    end
    q_diffs_plots = plot(1:length(q_diffs_dict["5_200"]), q_diffs_dict["5_200"], ylims = (0, 150000), fillcolor = :blue, linecolor = :blue, label = "T,I,B,W = 5, Volume = 200", legend = :topleft, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5, tickfontsize = 5, xlabel = "Episodes", ylabel = "Q-matrix Differences (T,I,B,W = 5)", title = "1 Step Q-matrix Policy Differences", right_margin = 18mm)
    plot!(1:length(q_diffs_dict["5_100"]), q_diffs_dict["5_100"], fillcolor = :red, linecolor = :red, label = "T,I,B,W = 5, Volume = 100", legend = :topleft, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5)
    plot!(1:length(q_diffs_dict["5_50"]), q_diffs_dict["5_50"], fillcolor = :green, linecolor = :green, label = "T,I,B,W = 5, Volume = 50", legend = :topleft, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5)
    subplot = twinx()
    plot!(subplot, 1:length(q_diffs_dict["10_200"]), q_diffs_dict["10_200"], ylims = (0, 13000), fillcolor = :magenta, linecolor = :magenta, label = "T,I,B,W = 10, Volume = 200", legend = :topright, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5, tickfontsize = 5, ylabel = "Q-matrix Differences (T,I,B,W = 10)")
    plot!(subplot, 1:length(q_diffs_dict["10_100"]), q_diffs_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = "T,I,B,W = 10, Volume = 100", legend = :topright, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5)
    plot!(subplot, 1:length(q_diffs_dict["10_50"]), q_diffs_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = "T,I,B,W = 10, Volume = 50", legend = :topright, fg_legend = :transparent, legendfontsize = 4, guidefontsize = 5)
    savefig(q_diffs_plots, path_to_files * "/Images/RL/QConvergence.pdf")

    # num_actions = Vector{Float64}()
    # num_trades = Vector{Float64}()
    # for i in 1:n
    #     push!(num_actions, l[i]["NumberActions"])
    #     push!(num_trades, l[i]["NumberTrades"])
    # end

    # p2 = plot(num_actions, fillcolor = :green, linecolor = :green, legend = false, xlabel = "Episodes", ylabel = "# Actions")
    # savefig(p2, path_to_files * "/Images/RL/NumberActionsConvergence.pdf")
   
    # p3 = plot(num_trades, fillcolor = :green, linecolor = :green, legend = false, xlabel = "Episodes", ylabel = "# Trades")
    # savefig(p3, path_to_files * "/Images/RL/NumberTradesConvergence.pdf")

    # # plot the initial actions and the final actions to see if there are differences in actions selected
    # l = data["Results_alpha0.1_iterations1000_V100_S10"]
    # actions1 = Vector{Float64}()
    # for action in l[1]["Actions"]
    #     push!(actions1, action)
    # end
    # pi1 = plot(1:l[1]["NumberActions"], getindex.(Ref(actionsMap), actions1) .* 100, seriestype = :line, fillcolor = :green, linecolor = :green, legend = false, xlabel = "Action Number", ylabel = "Volume Traded", title = "Iteration 1")
    # savefig(pi1, path_to_files * "/Images/RL/ActionsIteration1.pdf")
    # actionsN = Vector{Float64}()
    # n = 1000
    # for action in l[n]["Actions"]
    #     push!(actionsN, action)
    # end
    # piN = plot(1:l[n]["NumberActions"], getindex.(Ref(actionsMap), actionsN) .* 100, seriestype = :line, fillcolor = :green, linecolor = :green, legend = false, xlabel = "Action Number", ylabel = "Volume Traded", title = "Iteration " * string(n))
    # savefig(piN, path_to_files * "/Images/RL/ActionsIteration" * string(n) * ".pdf")

end
A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
actions = GenerateActions(A, maxVolunmeIncrease)
PlotRLConvergenceResults(actions)
#---------------------------------------------------------------------------------------------------

#----- State-action convergence -----# 
function StateActionConvergence(l::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, actionsMap::Dict)
    # TODO: Make file saving better with names
    n = length(l)

    # get max number of states (last iteration states)
    max_states = collect(keys(l[1000]["Q"]))

    # for each key get the policy over the iterations (if state does not exist then -1)
    actions_dict = Dict()
    for state in max_states
        actions = Vector{Float64}()
        for i in 1:n
            if state in collect(keys(l[i]["Q"]))
                push!(actions, actionsMap[argmax(l[i]["Q"][state])])
            else
                push!(actions, -1)
            end
        end
        push!(actions_dict, state => actions)
    end

    p = plot(actions_dict[max_states[1]], legend = false, xlabel = "Episodes", ylabel = "Actions", title = "State-Action Changes Over Time")
    for i in 2:length(max_states)
        plot!(actions_dict[max_states[i]])
    end
    savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V200_S10/StateActionConvergence_V200_S10.pdf")

end
# @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V200_S10.jld")["rl_results"]
# n = length(l)
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = 5                    # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
# I = 5                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
# B = 5                        # number of spread states
# W = 5                       # number of volume states
# StateActionConvergence(l, numT, I, B, W, A, actions)
#---------------------------------------------------------------------------------------------------

#----- Given a Q-matrix get the greedy policy -----# 
function GetPolicy(Q::Dict)
    P = Dict{Vector{Int64}, Int64}()
    for state in collect(keys(Q))
        push!(P, state => argmax(Q[state]))
    end
    return P 
end
#---------------------------------------------------------------------------------------------------

#----- Visualize the a single agents policy -----# 
function PolicyVisualization(Q::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, actionsMap::Dict)
    # TODO: Make file saving better with names
    P = GetPolicy(Q)
    plots = []
    inc = 1
    for s in B:-1:1# i in I:-1:1 # want volume to increase upwards in plot
        for v in 1:1:W # t in numT:-1:1 # want time remaining to decrease left to right
            # create a matrix that will store values for spread and volume states
            M = fill(0.0,B,W)
            s_counter = 1
            for i in 1:1:I # s in 1:1:B
                v_counter = 1
                for t in numT:-1:1 # v in 1:1:W
                    # for each of these states get the action associted with it, if it does not exist then -1
                    key = [t, i, s, v]
                    M[s_counter,v_counter] = -1
                    if key in collect(keys(P))
                        M[s_counter,v_counter] = actionsMap[P[key]]
                    end
                    v_counter += 1
                end
                s_counter += 1
            end
            # for a given t and i plot the actions taken over the spread and volume states
            xlabel = ""
            ylabel = ""
            if s == 5 && v == 5 # t == 5 && i == 5 specify the x and y labels for each individual heatmap
                xlabel = "Volume"
                ylabel = "Spread"
            end
            h = heatmap(1:B, 1:W, M, xlabel = xlabel, ylabel = ylabel, c = cgrad(:seismic, [0, 0.50, 0.78, 1]), clim = (-1, actionsMap[A]), guidefontsize = 4, tick_direction = :out, legend = false, tickfontsize = 4, margin = -1mm)
            # annotate!(h, [(j, i, text(M[i,j], 2,:black, :center)) for i in 1:B for j in 1:W])
            push!(plots, h)
        end
    end
    l = @layout[a{0.05w} grid(5,5); b{0.001h}]
    colorbar = heatmap([-1;getindex.(Ref(actionsMap), 1:A)].*ones(A+1,1), ylabel = "Inventory", ymirror = true, guidefontsize = 10, tickfontsize = 5, c = cgrad(:seismic, [0, 0.50, 0.78, 1]), legend=:none, xticks=:none, yticks=(1:1:(A+1), string.([-1;getindex.(Ref(actionsMap), 1:A)])), y_foreground_color_axis=:white, y_foreground_color_border=:white)
    empty = plot(title = "Time", titlefontsize = 10, legend=false,grid=false, foreground_color_axis=:white, foreground_color_border=:white, ticks = :none)
    p = plot(colorbar, plots..., empty, layout = l)
    savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V100_S5/PolicyPlot_V100_S5.pdf")

end
# @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V100_S5.jld")["rl_results"]
# n = length(l)
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = 5                    # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
# I = 5                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
# B = 5                        # number of spread states
# W = 5                       # number of volume states
# PolicyVisualization(l[1000]["Q"], numT, I, B, W, A, actions)
#---------------------------------------------------------------------------------------------------

#----- Agents Actions per State Value (averaged ove other states) -----# 
function AverageActionsPerStateValue(Q::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, actionsMap::Dict)
    # TODO: Make file saving better with names
    states = collect(keys(Q))

    # get average actions per time value
    avg_action_time = Vector{Float64}() # time remaining increases from start to finish
    for t in 1:numT
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[1] == t, states)]))
        push!(avg_action_time, mean(getindex.(Ref(actionsMap), action_ids)))
    end
    println()

    # get average actions per inventory value
    avg_action_inventory = Vector{Float64}() # time remaining increases from start to finish
    for i in 1:I
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[2] == i, states)]))
        push!(avg_action_inventory, mean(getindex.(Ref(actionsMap), action_ids)))
    end
    println()

    # get average actions per spread value
    avg_action_spread = Vector{Float64}() # time remaining increases from start to finish
    for s in 1:B
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[3] == s, states)]))
        push!(avg_action_spread, mean(getindex.(Ref(actionsMap), action_ids)))
    end
    println()

    # get average actions per volume value
    avg_action_volume = Vector{Float64}() # time remaining increases from start to finish
    for v in 1:W
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[4] == v, states)]))
        push!(avg_action_volume, mean(getindex.(Ref(actionsMap), action_ids)))
    end

    # plot the effects
    p = plot(reverse(avg_action_time), label = "time", legend = :bottomright)
    plot!(avg_action_inventory, label = "inventory")
    plot!(avg_action_spread, label = "spread")
    plot!(avg_action_volume, label = "volume")

    savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V100_S10/AverageActionEffects_V100_S10.pdf")

end
# @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V100_S10.jld")["rl_results"]
# n = length(l)
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = 10                    # number of time states (T must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
# I = 10                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
# B = 10                        # number of spread states
# W = 10                       # number of volume states
# AverageActionsPerStateValue(l[1000]["Q"], numT, I, B, W, A, actions)
#---------------------------------------------------------------------------------------------------

# l = load(path_to_files * "Data/RL/Training/Results0.1.jld")
# println(l["1"]["TotalReward"]) # 5.39204984e8
# println(l["100"]["TotalReward"]) # 5.99198636e8
# 5.58028675e8 (random 2, did not fully liquidate (9000)), 5.90548879e8 (random 4, fully liquidated)