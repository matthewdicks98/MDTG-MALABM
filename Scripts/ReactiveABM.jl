#=
ReactiveABM:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie, (some code was adapted from https://github.com/IvanJericevich/IJPCTG-ABMCoinTossX)
- Function: Functions used to simulate the event based ABM with a maximum of 1 RL selling agent
- Structure: 
    1. Market data listener (asynchronous method)
    2. Structures
    3. Agent specifications 
        i. High frequency liquidity providers
        ii. Chartists
        iii. Fundamentalists 
        iv. RL selling agent
    4. Rocket actor definitions
    5. Rocket subject definitions
    6. Updating of model management systems LOB
    7. Initialisation of the LOB
    8. Simulation
- Example:
    StartJVM()
    gateway = Login(1,1)
    seed = 1
    parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
    rlParameters = RLParameters(Nᵣₗ, initialQ, startTime, rlT, numT, V, Ntwap, I, B, W, A, actions, spread_states_df, volume_states_df, actionType, ϵ₀, discount_factor, α)
    print_and_plot = false                    
    write = false 
    rlTraders = true                        
    rlTraining = true                       
    @time mid_prices, micro_price, rl_result = simulate(parameters, rlParameters, gateway, rlTraders, rl_training, false, true, false, seed = seed, iteration = i)
    Logout(gateway)
- Prerequisites:
    1. CoinTossX is running
=#
ENV["JULIA_COPY_STACKS"]=1
using JavaCall, Rocket, Sockets, Random, Dates, Distributions, Plots, CSV, DataFrames, DataStructures, StatsBase
import Rocket.scheduled_next!

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/CoinTossXUtilities.jl")

#----- Listen for messages -----#
function Listen(receiver, messages_chnl, messages_received)
    try 
        while true
            s = Dates.now()
            message_loc = String(recv(receiver))
            # need to add when the message arrived
            message_loc_time = string(Dates.now()) * "|" * message_loc
            put!(messages_chnl, message_loc_time)
            push!(messages_received, message_loc_time)
            println("------------------- Port: " * message_loc_time) ########### Add back after testing
        end
    catch e
        if e isa EOFError
            println("\nEOF error caused by closing of socket connection\n")
        else
            println(e)
            @error "Something went wrong" exception=(e, catch_backtrace())
        end
    end
end
#---------------------------------------------------------------------------------------------------

#----- Auxilary Structures -----#
mutable struct Parameters
    Nᴸₜ::Int64 # Number of chartists for the low-frequency agent class
    Nᴸᵥ::Int64 # Number of fundamentalists for the low-frequency agent class
    Nᴴ::Int64 # Number of high-frequency agents
    δ::Float64 # Upper cut-off for LF agents decision rule
    κ::Float64 # Scaling factor for order placement depth
    ν::Float64 # Scaling factor for power law order size
    m₀::Float64 # Initial mid-price
    σᵥ::Float64 # Std dev in Normal for log-normal for fundamental value
    λmin::Float64 # min of the uniform dist for the forgetting factor of the chartist
    λmax::Float64 # max of the uniform dist for the forgetting factor of the chartist
    γ::Millisecond # the Time In Force (time the order stays in the order book until manually cancelled by the agent)
    T::Millisecond # Simulation time
	function Parameters(; Nᴸₜ = 5, Nᴸᵥ = 5, Nᴴ = 30, δ = 0.1 , κ = 3.5, ν = 5, m₀ = 10000, σᵥ = 0.015, λmin = 0.0005, λmax = 0.05, γ = Millisecond(1000), T = Millisecond(25000)) # 1000, 25000
		new(Nᴸₜ, Nᴸᵥ, Nᴴ, δ, κ, ν, m₀, σᵥ, λmin, λmax, γ, T)
	end
end 
mutable struct RLParameters
    Nᵣₗ::Int64              # total number of RL agents
    Nᵣₗ₁::Int64             # Number of RL agents, type 1
    Nᵣₗ₂::Int64             # Number of RL agents, type 2
    initialQsRL1::Vector{DefaultDict{Vector{Int64}, Vector{Float64}}} # the initial Q matrices from past iterations for type 2 RL agents 
    initialQsRL2::Vector{DefaultDict{Vector{Int64}, Vector{Float64}}} # the initial Q matrices from past iterations for type 2 RL agents 
    actionsRL1::OrderedDict{Int64, Float64} # stores the mapping from action index to actual action, action type 1
    actionsRL2::OrderedDict{Int64, Tuple{Float64, Float64}} # stores the mapping from action index to actual action, action type 2
    actionTypesRL1::Vector{String} # indicates type of MO the agents will perform, rl agent type 1
    actionTypesRL2::Vector{String} # indicates type of orders the agents will perform, rl agent type 2
    A1::Int64               # number of action states for agent type 1
    A2::Int64               # number of action states for agent type 2
    startTime::Millisecond # defines the time to start trading for the first execution (the rest of the executions can be defined from this and T)
    T::Millisecond         # total time to trade for a single execution of volume
    numT::Int64            # number of time states
    V::Int64               # total volume to trade for a single execution
    Ntwap::Float64         # number of shares to trade based on TWAP volume (V / numAgentDecisions)
    I::Int64               # number of inventory states
    B::Int64               # number of spread states
    W::Int64               # number of volume states
    spread_states_df::DataFrame # Stores the spread states for the RL agent
    volume_states_df::DataFrame # Stores the volume states for the RL agent
    ϵ::Float64             # used in epsilon greedy algorithm
    discount_factor::Float64 # used in Q update (discounts future rewards)
    α::Float64             # used in Q update
    λᵣ::Float64            # reward parameter used to control sensitivity to slippage
    γᵣ::Float64            # reward parameter used to control sensitivity to time 
end
mutable struct LimitOrder
    price::Int64
    volume::Int64
    trader::Symbol # traderMnemonic
end
mutable struct MovingAverage
    p̄ₜ::Float64 # General EWMA of mid-price
    actionTimes::Array{Millisecond,1} # Arrival process of when all agents makes a decision
    τ::Float64 # Time constant (used for EWMA computation) = Average inter-arrival of their decision times
end
mutable struct LOBState
    sₜ::Int64 # Current spread
    ρₜ::Float64 # Current order imbalance
    mₜ::Float64 # Current mid-price
    microPrice::Float64 # Current micro-price
	priceReference::Int64 # Price reference - last traded price
    bₜ::Int64 # Best bid
    aₜ::Int64 # Best ask
    bids::Dict{Int64, LimitOrder} # Stores all the active bids
    asks::Dict{Int64, LimitOrder} # Stores all the active asks
end
#---------------------------------------------------------------------------------------------------

#----- State Structure -----#
mutable struct SimulationState
    LOB::LOBState
    parameters::Parameters
    rlParameters::RLParameters
    gateway::TradingGateway
    initializing::Bool
    event_counter::Int64
    start_time::DateTime
    rl_traders_vec::Vector # stores the structs for all the RL agents in the model
    trade_vwap::Float64 # stores the vwap price for all the trades (not including RL agents)
    total_trade_volume::Int64 # stores the total traded volume (not including RL agents)
    total_price_volume::Int64 # Σpᵢvᵢ (not including RL agents)
end
#---------------------------------------------------------------------------------------------------

include(path_to_files * "Scripts/Actors.jl")
include(path_to_files * "Scripts/SimulationUtilities.jl") # have to include here otherwise it throws type errors (should refactor this)

#----- Define How Subject Passes Messages to Actors -----#
function nextState(subject::Subject, simulationstate::SimulationState)
    not_activated = Vector{Any}()
    for listener in subject.listeners
        push!(not_activated, listener)
    end
    while length(not_activated) > 0
        listener = rand(not_activated)
        scheduled_next!(listener.actor, simulationstate, listener.schedulerinstance)
        filter!(x -> x != listener, not_activated)
    end
end
#---------------------------------------------------------------------------------------------------

#----- Define Actor Actions -----#

# HF
Rocket.on_next!(actor::HighFrequency, simulationstate::SimulationState) = HighFrequencyAgentAction(actor, simulationstate)
Rocket.on_error!(actor::HighFrequency, err)      = error(err)
Rocket.on_complete!(actor::HighFrequency)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

# chartists
Rocket.on_next!(actor::Chartist, simulationstate::SimulationState) = ChartistAction(actor, simulationstate)
Rocket.on_error!(actor::Chartist, err)      = error(err)
Rocket.on_complete!(actor::Chartist)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

# fundamentalists
Rocket.on_next!(actor::Fundamentalist, simulationstate::SimulationState) = FundamentalistAction(actor, simulationstate)
Rocket.on_error!(actor::Fundamentalist, err)      = error(err)
Rocket.on_complete!(actor::Fundamentalist)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

# RL, type 1
Rocket.on_next!(actor::RL1, simulationstate::SimulationState) = RLAction1(actor, simulationstate)
Rocket.on_error!(actor::RL1, err)      = error(err)
Rocket.on_complete!(actor::RL1)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

# RL, type 2
Rocket.on_next!(actor::RL2, simulationstate::SimulationState) = RLAction2(actor, simulationstate)
Rocket.on_error!(actor::RL2, err)      = error(err)
Rocket.on_complete!(actor::RL2)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

#---------------------------------------------------------------------------------------------------

#----- Define Subject -----#

Rocket.on_next!(subject::Subject, simulationstate::SimulationState) = nextState(subject, simulationstate)

#---------------------------------------------------------------------------------------------------

#----- Update LOB state -----#
function UpdateLOBState!(simulationstate::SimulationState, message)

    # extract LOB from simulation state
    LOB = simulationstate.LOB

    # take the time out of the message
    msg = split(message, "|")[2:end]

	#msg = split(message, "|")
	fields = split(msg[1], ",")

    # need 2 types (Type is the original one and type is the one that changes (crossing LOs))
    Type = Symbol(fields[1]); type = Symbol(fields[1])
    side = Symbol(fields[2]); trader = Symbol(fields[3])

    # if the msg is of the form time|x,y,z| then there is no price or quantity executed so the order must be disgarded
    # Can happen if for example a MO is traded with no limit orders to eat it up, or cancelling an order that is not in the book anymore
    executions = msg[2:end]
    if executions == [""]
       return
    end

	for (i, execution) in enumerate(executions)
        type = Type
        executionFields = split(execution, ",")
        id = parse(Int, executionFields[1]); price = parse(Int, executionFields[2]); volume = parse(Int, executionFields[3])

        # If there is a trade where the traderMnemonic is HFx and there is an order id that is not in the orderbook then this is a new limit order
        # If the order is the first one in the executions list then the new LO is the same but the volume is the volume of that LO
        # less the rest of the volume in that order.
        # If it is the last in the executions then it is the same LO but the volume is the one in the executions
        if type == :Trade && side == :Buy # && fields[3][1:2] == "HF"
            # checks asks for a buy since this will be where the LO buys is executed against
            if !(id in keys(LOB.asks)) 
                # this is the excess order that needs to be a new limit order
                type = Symbol("New")
                
                # if it is the first then I need to remove some volume
                if i == 1
                    if length(executions[2:end]) == 0
                        println()
                        println("Message that cause the error: ")
                        println(message)
                        println()
                    end
                    volume = volume - sum(parse(Int, split(e, ",")[3]) for e in executions[2:end])
                end

            end
        elseif type == :Trade && side == :Sell # && fields[3][1:2] == "HF" 
            # checks bids for a sell since this will be where the LO buys is executed against
            if !(id in keys(LOB.bids))
                # this is the excess order that needs to be a new limit order
                type = Symbol("New")

                # if it is the first then I need to remove some volume
                if i == 1
                    if length(executions[2:end]) == 0
                        println()
                        println("Message that cause the error: ")
                        println(message)
                        println()
                    end
                    volume = volume - sum(parse(Int, split(e, ",")[3]) for e in executions[2:end])
                end

            end
        end

        if type == :New
            # don't want to push new limit orders that have 0 volume
            if volume > 0
                side == :Buy ? push!(LOB.bids, id => LimitOrder(price, volume, trader)) : push!(LOB.asks, id => LimitOrder(price, volume, trader))
            end
        elseif type == :Cancelled
            side == :Buy ? delete!(LOB.bids, -id) : delete!(LOB.asks, -id)
        elseif type == :Trade
            if side == :Buy
				LOB.priceReference = LOB.aₜ # price reference is the execution price of the previous trade (approx)
                LOB.asks[id].volume -= volume
                if LOB.asks[id].volume == 0
                    delete!(LOB.asks, id)
                end
            else
				LOB.priceReference = LOB.bₜ # price reference is the execution price of the previous trade (approx)
                LOB.bids[id].volume -= volume
                if LOB.bids[id].volume == 0
                    delete!(LOB.bids, id)
                end
            end
        end
    end
	totalBuyVolume = 0; totalSellVolume = 0
	if !isempty(LOB.bids) && !isempty(LOB.asks)
		LOB.bₜ = maximum(order -> order.price, values(LOB.bids)); LOB.aₜ = minimum(order -> order.price, values(LOB.asks))
		bidVolume = sum(order.volume for order in values(LOB.bids) if order.price == LOB.bₜ)
		askVolume = sum(order.volume for order in values(LOB.asks) if order.price == LOB.aₜ)
		LOB.microPrice = (LOB.bₜ * bidVolume + LOB.aₜ * askVolume) / (bidVolume + askVolume)
		totalBuyVolume = sum(order.volume for order in values(LOB.bids)); totalSellVolume = sum(order.volume for order in values(LOB.asks))
	else
		LOB.microPrice = NaN
		if !isempty(LOB.bids)
			LOB.bₜ = maximum(order -> order.price, values(LOB.bids))
			totalBuyVolume = sum(order.volume for order in values(LOB.bids))
		end
		if !isempty(LOB.asks)
			LOB.aₜ = minimum(order -> order.price, values(LOB.asks))
			totalSellVolume = sum(order.volume for order in values(LOB.asks))
		end
	end

	LOB.sₜ = abs(LOB.aₜ - LOB.bₜ)
	LOB.mₜ = (LOB.aₜ + LOB.bₜ) / 2
    LOB.ρₜ = (totalBuyVolume == 0) && (totalSellVolume == 0) ? 0.0 : (totalBuyVolume - totalSellVolume) / (totalBuyVolume + totalSellVolume)
end
#---------------------------------------------------------------------------------------------------

#----- Supplementary functions -----#
function PowerLaw(xₘ, α) # Volumes
    return xₘ / (rand() ^ (1 / α))
end
function ComputeAbmAgentsMarketVwap(simulationstate::SimulationState, message::String) # computes the markey VWAP for ABM agents (not including any RL agents)
    msg = split(message, "|")[2:end]
	fields = split(msg[1], ",")
    type = Symbol(fields[1]); trader = fields[3]

    if type == :New || occursin("RL", trader) # only compute trades and don't add RL traders trades to the VWAP volume
        return
    elseif type == :Trade
        executions = msg[2:end]
        if executions == [""]
            return
        end
        for (i, execution) in enumerate(executions)
            executionFields = split(execution, ",")
            price = parse(Int, executionFields[2]); volume = parse(Int, executionFields[3])
            simulationstate.trade_vwap = (1 / (simulationstate.total_trade_volume + volume)) * (simulationstate.total_price_volume + volume * price)
            simulationstate.total_trade_volume += volume
            simulationstate.total_price_volume += price * volume # Σpᵢvᵢ
        end
    end

end
function UpdateLimitOrderStates(rlAgents::Vector{Union{RL1,RL2}}, message::String)
    msg = split(message, "|")[2:end]
    fields = split(msg[1], ",")
    trader = fields[3]
    type = fields[1]
    executions = msg[2:end]
    orderId = parse(Int, fields[4])

    if type == "New" || executions == [""]
        return
    end

    for (i, execution) in enumerate(executions)
        executionFields = split(execution, ",")
        id = parse(Int, executionFields[1]); price = parse(Int, executionFields[2]); volume = parse(Int, executionFields[3])
        for rlAgent in rlAgents
            if (id in collect(keys(rlAgent.currentLOs))) && (orderId != id) && (id > 0) # find if match with RL agents orders, not limit order, not cancellation
                rlAgent.currentLOs[id]["matched_volume"] += volume
                break
            end
            if (-id in collect(keys(rlAgent.currentLOs))) && (id < 0) # cancellations
                rlAgent.currentLOs[-id]["status"] = "cancelled"
                break
            end
        end
    end

end

function UpdateMarketOrderStates(rlAgents::Vector{Union{RL1,RL2}}, message::String)
    # println()
    # take the time out of the message
    msg = split(message, "|")[2:end]
    fields = split(msg[1], ",")
    trader = fields[3]
    orderId = parse(Int, fields[4])
    traderId = parse(Int, trader[3:end])
    executions = msg[2:end]
    if executions == [""] # should never occur since market order but just in case
       return
    end

    if rlAgents[traderId].agentType == "Type2"
        if orderId in collect(keys(rlAgents[traderId].currentLOs)) # if the market orders id is in the traders LOs not market orders
            # treat as if the LO has been matched
            for execution in executions
                executionFields = split(execution, ",")
                id = parse(Int, executionFields[1]); price = parse(Int, executionFields[2]); volume = parse(Int, executionFields[3])
                if orderId != id
                    rlAgents[traderId].currentLOs[orderId]["matched_volume"] += volume
                end
            end
            # println(rlAgents[traderId].currentLOs[orderId])
        else # pure market order message
            rlAgents[traderId].currentMOs[orderId]["trade_message"] = message
            # println(rlAgents[traderId].currentMOs[orderId])
        end
    elseif rlAgents[traderId].agentType == "Type1"
        rlAgents[traderId].currentMOs[orderId]["trade_message"] = message
    end

    # println(message)
    # println()
end
#---------------------------------------------------------------------------------------------------

#----- Initialize LOB -----#
function InitializeLOB(simulationstate::SimulationState, messages_chnl::Channel, source::Subject, number_initial_messages::Int64, initial_messages_received::Vector{String}, messages_received::Vector{String})

    # always leave a message in the channel so that when the sim starts agents have an event to trade off
    
    while true

        # send the event to be processed by the actors
        if simulationstate.event_counter <= number_initial_messages
            # ask the hf agents to trade
            Rocket.next!(source, simulationstate)
        end

        # Sleep the main task for a tiny amount of time to switch to the listening task
        sleep(0.05) # 1 microsecond

        if isready(messages_chnl) && simulationstate.event_counter <= number_initial_messages # if there are messages in the channel then take the last update

            while isready(messages_chnl)
                message = take!(messages_chnl)

                # push to the initial messages array
                push!(initial_messages_received, message)

                # use the messages_recieved array to keep track of the number of messages in the channel
                popfirst!(messages_received)

                # update the LOBState with new order 
                UpdateLOBState!(simulationstate, message)

            end 

        elseif isready(messages_chnl) && simulationstate.event_counter > number_initial_messages

            while isready(messages_chnl)
                message = take!(messages_chnl)

                # push to the initial messages array
                push!(initial_messages_received, message)

                # use the messages_recieved array to keep track of the number of messages in the channel
                popfirst!(messages_received)

                # update the LOBState with new order 
                UpdateLOBState!(simulationstate, message)

                # if there is only 1 message in the array then there is 1 message in the channel so break the initialization
                # also clear the array so that we only keep the messages received from after the initialization
                if length(messages_received) == 1
                    message = popfirst!(messages_received)
                    push!(initial_messages_received, message)
                    break
                end

            end 

            break # break out of the initialization loop

        end

    end

    return false

end
#---------------------------------------------------------------------------------------------------

#----- Simulate the event based ABM (1 RL selling agent can be added) -----#
function simulate(parameters::Parameters, rlParameters::RLParameters, gateway::TradingGateway, rlTraders::Bool, rlTraining::Bool, print_and_plot::Bool, write_messages::Bool, write_volume_spread::Bool; seed::Int64, iteration::Int64)

    initial_messages_received = Vector{String}() # stores all initialization messages
    messages_received = Vector{String}()         # stores all messages after initialization

    # start new LOB (trading session)
    StartLOB(gateway) 

    # open the channel that stores all the messages read from the UDP buffer
    chnl_size = Inf
    messages_chnl = Channel(chnl_size)

    # initialize LOBState (used to get agents subscribed)
    LOB = LOBState(40, 0, parameters.m₀, NaN, parameters.m₀, parameters.m₀ - 20, parameters.m₀ + 20, Dict{Int64, LimitOrder}(), Dict{Int64, LimitOrder}())

    # Set the first subject messages
    simulationstate = SimulationState(LOB, parameters, rlParameters, gateway, true, 1, Dates.now(), Vector{Union{RL1,RL2}}(), 0, 0, 0)

    # open the UDP socket
    receiver = UDPSocket()
    connected = bind(receiver, ip"127.0.0.1", 1234)
    task = @async Listen(receiver, messages_chnl, messages_received)

    # initialize the traders
    hf_traders_vec = map(i -> HighFrequency(i, "HF"*string(i), Array{Millisecond,1}(), Array{Tuple{DateTime, Order}, 1}()), 1:parameters.Nᴴ)

    # set the seed to ensure reproducibility (for chartist forgetting factors)
    Random.seed!(seed)
    char_traders_vec = map(i -> Chartist(i, "TF"*string(i), parameters.m₀, Array{Millisecond,1}(), rand(Uniform(parameters.λmin, parameters.λmax))), 1:parameters.Nᴸₜ)
    
    # set the seed to ensure reproducibility (for fundamental prices)
    Random.seed!(seed)
    fun_traders_vec = map(i -> Fundamentalist(i, "VI"*string(i), parameters.m₀ * exp(rand(Normal(0, parameters.σᵥ))), Array{Millisecond,1}()), 1:parameters.Nᴸᵥ)

    # initialize RL Traders (change to deal with multiple RL agents having different types)
    if rlTraders
        # make a copy of the initial Q and add RL type 1 agents
        for i in 1:rlParameters.Nᵣₗ₁
            Q = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A1))
            for key in keys(rlParameters.initialQsRL1[i])
                Q[key] = copy(rlParameters.initialQsRL1[i][key])
            end
            rl_trader = RL1(i, "RL"*string(i), "Type1", Array{Millisecond,1}(), Vector{Int64}(), rlParameters.actionTypesRL1[i], false, rlParameters.T.value, rlParameters.V, false, Vector{Float64}(), Q, Vector{Int64}(), OrderedDict(), 0, 0, 0) 
            push!(simulationstate.rl_traders_vec, rl_trader)
        end
        # make a copy of the initial Q and add RL type 2 agents
        for i in 1:rlParameters.Nᵣₗ₂
            Q = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A2))
            for key in keys(rlParameters.initialQsRL2[i])
                Q[key] = copy(rlParameters.initialQsRL2[i][key])
            end
            # will need to change as RL agent type 2 has different storage
            rl_trader = RL2(Nᵣₗ₁ + i, "RL"*string(Nᵣₗ₁ + i), "Type2", Array{Millisecond,1}(), Vector{Int64}(), rlParameters.actionTypesRL2[i], false, rlParameters.T.value, rlParameters.V, false, Vector{Float64}(), Q, Vector{Int64}(), OrderedDict(), OrderedDict(), Set{Int64}(), 0, 0, 0) 
            push!(simulationstate.rl_traders_vec, rl_trader)
        end
    end

    # initialize the Subject and subscribe actors to it
    source = Subject(SimulationState)
    map(i -> subscribe!(source, i), hf_traders_vec)
    map(i -> subscribe!(source, i), char_traders_vec)
    map(i -> subscribe!(source, i), fun_traders_vec)
    if rlTraders
        map(i -> subscribe!(source, i), simulationstate.rl_traders_vec)
    end

    # storage for the prices
    mid_prices = Array{Float64, 1}()
    micro_prices = Array{Float64, 1}()

    # define some storage for test images
    running_totals = nothing
    if print_and_plot || write_volume_spread
        running_totals = InitializeRunningTotals(parameters.Nᴸₜ, parameters.Nᴸᵥ)
    end

    # initialize LOBState (generate a bunch of limit orders from the HF traders that will be used as the initial state before the trading starts)
    println()
    println("\n#################################################################### Initialization Started\n")

    # global initializing = true
    number_initial_messages = 1001
    simulationstate.initializing = InitializeLOB(simulationstate, messages_chnl, source, number_initial_messages, initial_messages_received, messages_received) # takes about 3.2 seconds

    # push start state info to the prices
    push!(mid_prices, simulationstate.LOB.mₜ)
    push!(micro_prices, simulationstate.LOB.microPrice)

    # push start state info to the running totals
    if print_and_plot || write_volume_spread
        UpdateRunningTotals(running_totals, parameters.Nᴸₜ, parameters.Nᴸᵥ, simulationstate.LOB.bₜ, simulationstate.LOB.aₜ, char_traders_vec, fun_traders_vec, simulationstate.LOB.ρₜ, simulationstate.LOB.sₜ, simulationstate.LOB.asks, simulationstate.LOB.bids)
    end

    println("\n#################################################################### Initialization Done\n")
    println()

    #----- Event Loop -----#

    # get the current time to use for while loop (causes the first HF trade to be a bit before the LF traders but after it seems fine)
    simulationstate.start_time = Dates.now()

    try
        @time while true
            
            # Sleep the main task for a tiny amount of time to switch to the listening task
            sleep(0.055) # tune to make 430

            # send the event to be processed by the actors
            if isready(messages_chnl) # if there are messages in the channel then take the last update

                # update the LOB with all the new events
                while isready(messages_chnl)

                    message = take!(messages_chnl)
                    
                    # update the LOBState with new order 
                    UpdateLOBState!(simulationstate, message)

                    # if the message was from an RL trader then add this to the RL messages Vector, update the market VWAPs, and update LO states
                    if rlTraders

                        # compute the market VWAP for the ABM agents but not the RL agents
                        ComputeAbmAgentsMarketVwap(simulationstate, message)

                        # take the time out of the message
                        msg = split(message, "|")[2:end]
                        fields = split(msg[1], ",")
                        trader = fields[3]
                        type = fields[1]

                        # update RL type 2 traders limit order states (assumes type 1 agents are always initialised before type 2)
                        if Nᵣₗ₂ > 0
                            UpdateLimitOrderStates(simulationstate.rl_traders_vec[(Nᵣₗ₁+1):end], message)
                        end

                        # store the traders rl market order messages 
                        if occursin("RL", trader) && type == "Trade"
                            UpdateMarketOrderStates(simulationstate.rl_traders_vec, message)
                        end

                    end

                    # Update running prices
                    push!(mid_prices, simulationstate.LOB.mₜ)
                    push!(micro_prices, simulationstate.LOB.microPrice)

                    # Update running state info
                    if print_and_plot || write_volume_spread
                        UpdateRunningTotals(running_totals, parameters.Nᴸₜ, parameters.Nᴸᵥ, simulationstate.LOB.bₜ, simulationstate.LOB.aₜ, char_traders_vec, fun_traders_vec, simulationstate.LOB.ρₜ, simulationstate.LOB.sₜ, simulationstate.LOB.asks, simulationstate.LOB.bids)
                    end

                end

                # check if the actors want to act based on the updated state
                Rocket.next!(source, simulationstate)

            else
                break
            end
        end
    catch e
        println(e)
        # Close the channel
        close(messages_chnl)
        # close the Socket (can generate and EOF error in the async task at the end of the sim)
        close(receiver)
        # clear LOB and end trading session
        EndLOB(gateway)
        # print the error
        @error "Something went wrong" exception=(e, catch_backtrace())
        # return nothing so sensitivity analysis and calibration can stopped
        if rlTraders
            return nothing, nothing, nothing
        else
            return nothing, nothing
        end
    end
    #---------------------------------------------------------------------------------------------------

    # Close the channel
    close(messages_chnl)

    # close the Socket (can generate and EOF error in the async task at the end of the sim)
    close(receiver)

    # clear LOB and end trading session
    EndLOB(gateway)

    # used to ensure that there is not too much variability in the messages recieved (occured when I kept the semaphore)
    println()
    println("Messages Received ", length(messages_received))
    println()

    # Print summary stats and plot test images 
    if print_and_plot
        SummaryAndTestImages(messages_chnl, parameters, simulationstate.LOB, mid_prices, running_totals.best_bids, running_totals.best_asks, running_totals.spreads, running_totals.imbalances, running_totals.chartist_ma, running_totals.fundamentalist_f, running_totals.ask_volumes, running_totals.bid_volumes, hf_traders_vec, char_traders_vec, fun_traders_vec, messages_received, running_totals.best_bid_volumes, running_totals.best_ask_volumes)
    end

    # write all the orders received after initialization to a file 
    if write_messages
        WriteMessages(initial_messages_received, messages_received, rlTraining, iteration)
    end

    # write volume and spread information that create historical spread and volume distribtutions
    if write_volume_spread
        WriteVolumeSpreadData(running_totals.spreads, running_totals.bid_volumes, running_totals.best_bid_volumes, running_totals.ask_volumes, running_totals.best_ask_volumes)
    end

    if rlTraders
        for i in 1:length(simulationstate.rl_traders_vec)
            println()
            println("RL Agent " * string(i) * ":")
            println("Agent Type: ", simulationstate.rl_traders_vec[i].agentType)
            println("Action Type: " * simulationstate.rl_traders_vec[i].actionType)
            println("Number of Actions = ", length(simulationstate.rl_traders_vec[i].actions))
            println("Return = ", sum(simulationstate.rl_traders_vec[i].R))
            println("Average reward = ", mean(simulationstate.rl_traders_vec[i].R))
            println("Min Reward = ", sort(simulationstate.rl_traders_vec[i].R)[1])
            println("Second min Reward = ", sort(simulationstate.rl_traders_vec[i].R)[2])
            println("Max Reward = ", sort(simulationstate.rl_traders_vec[i].R)[end])
            println("Remaining ineventory = ", simulationstate.rl_traders_vec[i].i)
            println()
        end
    end

    # return mid-prices and micro-prices, and if rl traded then return the Q matrices and rewards for the simulations
    if rlTraders # extend for multiple rl agents
        rl_result = Dict()
        map(i -> push!(rl_result, "rlAgent_" * string(i) => Dict("Q" => simulationstate.rl_traders_vec[i].Q, "Rewards" => simulationstate.rl_traders_vec[i].R, "TotalReward" => sum(simulationstate.rl_traders_vec[i].R), "Actions" => simulationstate.rl_traders_vec[i].actions, "NumberActions" => length(simulationstate.rl_traders_vec[i].actions), "NumberTrades" => length(simulationstate.rl_traders_vec[i].actionTimes), "ActionType" => simulationstate.rl_traders_vec[i].actionType, "AgentType" => simulationstate.rl_traders_vec[i].agentType, "RemainingInventory" => simulationstate.rl_traders_vec[i].i)), 1:length(simulationstate.rl_traders_vec))
        return mid_prices, micro_prices, rl_result
    else
        return mid_prices, micro_prices
    end

end
#---------------------------------------------------------------------------------------------------