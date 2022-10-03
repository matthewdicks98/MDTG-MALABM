#=
RLUtilities:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie
- Function: Provide functions that can test and train a single RL selling agent in an event based ABM
- Structure:
    1. Simulate historical distributions for spread and volume states (ensure CoinTossX is started)
    2. Create spread and volume state spaces from hisorical distributions
    3. Generate action spaced
    4. Returning of current state
    5. Test a single RL agent in an ABM for 1 iteration
    6. Train 1 RL selling agent
- Examples:
    1. Simulate historical distributions for spread and volume states (ensure CoinTossX is started)
        parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
        GenerateHistoricalDistributions(parameters, 365)
    2. Create spread and volume state spaces from hisorical distributions
        spread_states_df, volume_states_df = HistoricalDistributionsStates(5,5,false,false,true,1) 
    3. Train 1 RL selling agent
        parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
        rlParameters = RLParameters(Nᵣₗ, initialQ, startTime, rlT, numT, V, Ntwap, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, ϵ₀, discount_factor, α)
        numEpisodes = 1000
        steps = [75, 175, 50, 100]    
        stepSizes = [0.1, 0.8, 0.09, 0]   
        iterationsPerWrite = 100
        print_and_plot = false                    
        write_messages = false                            
        write_volume_spread = false
        rlTraders = true
        @time TrainRL(parameters, rlParameters, numEpisodes, rlTraders, iterationsPerWrite, steps, stepSizes) [takes about 8hrs]
=#
ENV["JULIA_COPY_STACKS"]=1
using DataFrames, CSV, Plots, Statistics, DataStructures, JLD, Plots.PlotMeasures

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/ReactiveABM.jl"); include(path_to_files * "Scripts/CoinTossXUtilities.jl")

#----- Simulate historical distributions for spread and volume states -----# (ensure that CoinTossX has started)
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
# set the ABM parameters
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
# spread_states_df, volume_states_df = HistoricalDistributionsStates(5,5,false,false,true,1)
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

    # println("Trade message: ", trade_message)
    # println("Total volume traded: ", total_volume_traded)
    # println("Sum price volume: ", sum_price_volume)
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
    # println("Spread: ", LOB.sₜ)
    # println("Spread state: ", sₙ)
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
    # println("Volume: ", v)
    # println("Volume state: ", vₙ)
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
    # println("Time remaining: ", t)
    # println("Time state: ", tₙ)
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
    # println("Inventory remaining: ", i)
    # println("Inventory state: ", iₙ)
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
    # want to maximize the profit or maximise the negative of the cost 
    a_star = argmax(Q[state]) 
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
    γ = Millisecond(1000) # fixed at 25000
    T = Millisecond(25000) # fixed at 25000 
    seed = 1 # 6, 8, 9

    parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

    # Rl parameters
    Nᵣₗ = 1                      # num rl agents
    startTime = Millisecond(0)   # start time for RL agents (keep it at the start of the sim until it is needed to have multiple)
    rlT = Millisecond(24500)     # 24500 execution duration for RL agents (needs to ensure that RL agents finish before other agents to allow for correct computation of final cost)
    numT = 5                    # number of time states (rlT must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
    V = 43000 # 43000                    # (266/2 * 450) volume to trade in each execution (ensure it is large enough so that price impact occurs at higher TWAP volumes and lower TWAP volumes no price impact)
    I = 5                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
    B = 5                        # number of spread states
    W = 5                        # number of volume states
    A = 9                       # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
    maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)

    # reward function parameters
    λᵣ = 0.003                   # controls sensitivity to Slippage
    γᵣ = 0.25                    # controls sensitivity to time

    spread_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/SpreadStates1_S5.csv") |> DataFrame
    volume_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/VolumeStates1_S5.csv") |> DataFrame
    # spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,true,1)
    actions = GenerateActions(A, maxVolunmeIncrease)
    actionType = "Sell"
    buyP = nothing # Using it as c now
    ϵ₀ = 1            # used in epsilon greedy algorithm
    discount_factor = 1 # used in Q update (discounts future rewards)
    α = 0.1             # used in Q update (α = 0.1, 0.01, 0.5)
    initialQ = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, A))
    numDecisions = 430 # 430 each agent has approx 430 decisions to make per simulation, Ntwap = V / numDecisions (this is fixed, but need to get estimated for new hardware)
    Ntwap = V / numDecisions

    rlParameters = RLParameters(Nᵣₗ, initialQ, startTime, rlT, numT, V, Ntwap, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, buyP, ϵ₀, discount_factor, α, λᵣ, γᵣ)

    # rl training parameters
    # numEpisodes = 2
    # steps = [75, 175, 50, 100]    # number of steps for each percentage decrease
    # stepSizes = [0.1, 0.8, 0.09, 0]   # Percentage decrease over the number of steps
    # iterationsPerWrite = 100

    # set the parameters that dictate output
    print_and_plot = false                    # Print out useful info about sim and plot simulation time series info
    write_messages = false                             # Says whether or not the messages data must be written to a file
    write_volume_spread = false
    rlTraders = true

    # run the simulation
    StartJVM()
    gateway = Login(1,1)
    try 
        @time simulate(parameters, rlParameters, gateway, rlTraders, false, print_and_plot, write_messages, write_volume_spread, seed = seed, iteration = 1)
    catch e
        @error "Something went wrong" exception=(e, catch_backtrace())
    finally
        Logout(gateway)
    end
end
# TestRunRLABM()
#---------------------------------------------------------------------------------------------------

#----- Train 1 RL selling agent in an ABM -----# 
function TrainRL(parameters::Parameters, rlParameters::RLParameters, numEpisodes::Int64, rlTraders::Bool, iterationsPerWrite::Int64, steps::Vector{Int64}, stepSizes::Vector{Float64})

    println("-------------------------------- Training RL Agent --------------------------------")

    rl_training = true

    # set storage for single agent rl_results (change when multiple)
    rl_results = Dict{Int64,Dict}()

    # store the prev q for checking
    prev_qs = map(i -> DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A)), 1:rlParameters.Nᵣₗ)
    prev_cs = map(i -> DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A)), 1:rlParameters.Nᵣₗ)

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

            # determine if you need to write the messages (always write first and last) CHANGE TO MAKE SURE IT WRITES AFTER TESTING
            if i % iterationsPerWrite == 0 || i == numEpisodes || i == 1
                @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, rlTraders, rl_training, false, true, false, seed = seed, iteration = i)
            else
                @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, rlTraders, rl_training, false, false, false, seed = seed, iteration = i)
            end           

            @assert((!isnothing(mid_prices)) && (!isnothing(micro_price)) && (!isnothing(rl_result)))

            # update ϵ
            rlParameters.ϵ = max(rlParameters.ϵ - (ϵ₀ - (1 - stepSizes[steps_counter]) * ϵ₀)/steps[steps_counter], 0)

            if i == sum(steps[1:steps_counter]) # move to the next step regime
                steps_counter += 1
            end

            println()
            println("Number of Trades: ")
            for j in 1:rlParameters.Nᵣₗ
                println("RL Agent " * string(j) * " = " * string(rl_result["rlAgent_" * string(j)]["NumberTrades"]))
            end
            println()

            for j in 1:rlParameters.Nᵣₗ
    
                # some printing for testing
                # println("Prev C ", j)
                # for key in keys(prev_cs[j])
                #     println(key, " => ", prev_cs[j][key])
                # end
    
                # println("Result C ", j)
                # for key in keys(rl_result["rlAgent_" * string(j)]["C"])
                #     println(key, " => ", rl_result["rlAgent_" * string(j)]["C"][key])
                # end

                # actual storage updating
                if i > 1                
                    if prev_qs[j] != rlParameters.initialQs[j]
                        println("Q Error")
                        return
                    end
                end
                if i > 1                
                    if prev_cs[j] != rlParameters.initialCs[j]
                        println("C Error")
                        return
                    end
                end


                # used to test the passing of Q from one simulation to another
                for key in keys(rl_result["rlAgent_" * string(j)]["Q"])
                    prev_qs[j][key] = copy(rl_result["rlAgent_" * string(j)]["Q"][key])
                    prev_cs[j][key] = copy(rl_result["rlAgent_" * string(j)]["C"][key])
                end

                # for the next simulation use the learnt Q from the previous as the new Q (make a copy, copy is made in the simulation)
                rlParameters.initialQs[j] = rl_result["rlAgent_" * string(j)]["Q"]
                rlParameters.initialCs[j] = rl_result["rlAgent_" * string(j)]["C"]

            end

            # save the rl_result to the vector
            for j in 1:rlParameters.Nᵣₗ
                rl_result["rlAgent_" * string(j)]["Q"] = Dict(rl_result["rlAgent_" * string(j)]["Q"])
                rl_result["rlAgent_" * string(j)]["C"] = Dict(rl_result["rlAgent_" * string(j)]["C"])
            end
            push!(rl_results, i => rl_result)

            # garbage collect
            GC.gc()
        end
    catch e
        @error "Something went wrong" exception=(e, catch_backtrace())
    finally
        Logout(gateway)
        StopCoinTossX()
        # write results to a file
        @time save(path_to_files * "/Data/RL/Training/ResultsTest.jld", "rl_results", rl_results)
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
γ = Millisecond(1000) # fixed at 25000
T = Millisecond(25000) # fixed at 25000 
seed = 1 # 6, 8, 9

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

# Rl parameters
Nᵇᵣₗ = 1                       # num rl buying agents
Nˢᵣₗ = 0                       # num rl selling agents
Nᵣₗ = Nᵇᵣₗ + Nˢᵣₗ             # num rl agents
startTime = Millisecond(0)   # start time for RL agents (keep it at the start of the sim until it is needed to have multiple)
rlT = Millisecond(24500)     # 24500 execution duration for RL agents (needs to ensure that RL agents finish before other agents to allow for correct computation of final cost)
numT = 5                    # number of time states (rlT must be divisible by numT to ensure evenly spaced intervals, error will be thrown) (not including zero state, for negative time)
V = 43000                    # (266/2 * 450) volume to trade in each execution (ensure it is large enough so that price impact occurs at higher TWAP volumes and lower TWAP volumes no price impact)
I = 5                       # number of invetory states (I must divide V to ensure evenly spaced intervals, error will be thrown) (not including terminal state)
B = 5                        # number of spread states
W = 5                        # number of volume states
A = 9                       # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)

# reward function parameters
λᵣ = 0.003                   # controls sensitivity to Slippage
γᵣ = 0.25                    # controls sensitivity to time

spread_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/SpreadStates1_S5.csv") |> DataFrame
volume_states_df = CSV.File(path_to_files * "/Data/RL/SpreadVolumeStates/VolumeStates1_S5.csv") |> DataFrame
# spread_states_df, volume_states_df = HistoricalDistributionsStates(B,W,false,false,true,1)
actions = GenerateActions(A, maxVolunmeIncrease)
actionTypes = append!(["Buy" for i in 1:Nᵇᵣₗ], ["Sell" for i in 1:Nˢᵣₗ])
ϵ₀ = 1            # used in epsilon greedy algorithm
discount_factor = 1 # used in Q update (discounts future rewards)
α = 0.1             # used in Q update (α = 0.1, 0.01, 0.5)
initialQs = map(i -> DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, A)), 1:Nᵣₗ)
initialCs = map(i -> DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, A)), 1:Nᵣₗ) # might need to remove if memory becomes an issue
numDecisions = 430 # 430 each agent has approx 430 decisions to make per simulation, Ntwap = V / numDecisions (this is fixed, but need to get estimated for new hardware)
Ntwap = V / numDecisions

rlParameters = RLParameters(Nᵣₗ, initialQs, initialCs, startTime, rlT, numT, V, Ntwap, I, B, W, A, actions, spread_states_df, volume_states_df, actionTypes, ϵ₀, discount_factor, α, λᵣ, γᵣ)

# rl training parameters
numEpisodes = 1000 # 1000
steps = [200, 400, 150, 250]  # number of steps for each percentage decrease [200, 400, 150, 250] [40, 80, 30, 50]
stepSizes = [0.1, 0.8, 0.09, 0]   # Percentage decrease over the number of steps
iterationsPerWrite = 10

# set the parameters that dictate output
print_and_plot = false                    # Print out useful info about sim and plot simulation time series info
write_messages = false                             # Says whether or not the messages data must be written to a file
write_volume_spread = false
rlTraders = true

# train the agent
@time TrainRL(parameters, rlParameters, numEpisodes, rlTraders, iterationsPerWrite, steps, stepSizes)

#---------------------------------------------------------------------------------------------------
 
