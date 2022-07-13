ENV["JULIA_COPY_STACKS"]=1
using JavaCall, Rocket, Sockets, Random, Dates, Distributions, Plots, CSV, DataFrames, DataStructures, StatsBase
import Rocket.scheduled_next!

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/CoinTossXUtilities.jl")

#----- Listen for messages -----#
function Listen(receiver, messages_chnl, messages_received)
    try 
        while true
            message_loc = String(recv(receiver))
            # need to add when the message arrived
            message_loc_time = string(Dates.now()) * "|" * message_loc
            put!(messages_chnl, message_loc_time)
            push!(messages_received, message_loc_time)
            # println("------------------- Port: " * message_loc_time) # keep for testing ############################## Add Print back here and to client in CoinTossX
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
    Nᵣₗ::Int64             # Number of RL agents
    initialQ::DefaultDict{Vector{Int64}, Vector{Float64}} # the initial Q matrix from past iterations (change when there are multiple agents)
    startTime::Millisecond # defines the time to start trading for the first execution (the rest of the executions can be defined from this and T)
    T::Millisecond         # total time to trade for a single execution of volume
    numT::Int64            # number of time states
    V::Int64               # total volume to trade for a single execution
    Ntwap::Float64         # number of shares to trade based on TWAP volume (V / numAgentDecisions)
    I::Int64               # number of inventory states
    B::Int64               # number of spread states
    W::Int64               # number of volume states
    A::Int64               # number of action states
    actions::Dict{Int64, Float64} # stores the mapping from action index to actual action
    spread_states_df::DataFrame # Stores the spread states for the RL agent
    volume_states_df::DataFrame # Stores the volume states for the RL agent
    type::String           # indicates type of MO the agent will perform if it is a single agent 
    ϵ::Float64             # used in epsilon greedy algorithm
    discount_factor::Float64 # used in Q update (discounts future rewards)
    α::Float64             # used in Q update
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
    rlMessages::Vector{String} # stores the RL execution messages used for the reward and inventory counts
    gateway::TradingGateway
    initializing::Bool
    event_counter::Int64
    start_time::DateTime
end
#---------------------------------------------------------------------------------------------------

#----- Agent Structures -----# (all actors accept the string message as input)
mutable struct Chartist <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    p̄ₜ::Float64 # Agent's mid-price EWMA
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a decision
    λ::Float64 # forgetting factor
end
mutable struct Fundamentalist <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    fₜ::Float64 # Current perceived value
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a decision
end
mutable struct HighFrequency <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a trade
    currentOrders::Array{Tuple{DateTime, Order}, 1} # the HF agents current orders in the order book (used for cancellations) (DateTime is the time the order was sent)
end

mutable struct RL <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a decision
    trade_messages::Vector{String}    # store all the messages when the agent made a trade
    actions::Vector{Int64}            # store all the actions the agent took

    actionType::String     # defines if the agent will be buying or selling
    
    t::Int64               # time remaining
    i::Int64               # inventory remaining/getting             
    done::Bool             # indicates if the trader has traded all the volume

    R::Vector{Float64}   # rewards stored over the course of a single execution
    Q::DefaultDict       # is the Q matrix for the RL agent for a single execution (only add a state-action pair when it is seen for the first time, don't initialize full Q)

    prev_state::Vector{Int64}     # stores the previous state for updating of Q
    prev_action::Int64          # stores the previous action for updating of Q
end
#---------------------------------------------------------------------------------------------------

include(path_to_files * "Scripts/SimulationUtilities.jl") # have to include here otherwise it throws type errors (should refactor this)

#----- Agent rules -----# 
function HighFrequencyAgentAction(highfrquency::HighFrequency, simulationstate::SimulationState)
    # TODO: Cancellations

    # do not trade if trading time is finished
    if !(simulationstate.initializing)
        if !(Dates.now() - simulationstate.start_time < simulationstate.parameters.T)
            return
        end
    end

    # cancel orders that have not been matched (dont have to account for the order being in the book or not since an empty message is sent back from CTX)
    if !(simulationstate.initializing) && length(highfrquency.currentOrders) > 0 # dont cancell orders during initialization and make sure there are orders to cancel
        # check if oldest order needs to be cancelled
        if Dates.now() - highfrquency.currentOrders[1][1] > simulationstate.parameters.γ

            # find all orders that need to be cancelled
            timed_out_inds = findall(x -> Dates.now() - x[1] > simulationstate.parameters.γ, highfrquency.currentOrders)

            # get all the orders that are in the LOB
            cancel_inds = Vector{Int64}()
            for ind in timed_out_inds
                if (highfrquency.currentOrders[ind][2].orderId in keys(simulationstate.LOB.bids)) || (highfrquency.currentOrders[ind][2].orderId in keys(simulationstate.LOB.asks))
                    push!(cancel_inds, ind)
                end
            end

            # send cancellation orders through
            for ind in cancel_inds
                CancelOrder(simulationstate.gateway, highfrquency.currentOrders[ind][2])
            end
            
            # delete the orders from the currentOrders array
            deleteat!(highfrquency.currentOrders, cancel_inds)

        end
    end

    order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("HF", highfrquency.traderId), type = "Limit")

    θ = simulationstate.LOB.ρₜ/2 + .5 # Probability of placing an ask
    order.side = rand() < θ ? "Sell" : "Buy"
    if order.side == "Sell"
        α = 1 - (simulationstate.LOB.ρₜ/simulationstate.parameters.ν) # Shape for power law

        # if spread is 0 set η = 0
        if simulationstate.LOB.sₜ == 0
            η = 0
        else
            η = floor(rand(Gamma(simulationstate.LOB.sₜ, exp(simulationstate.LOB.ρₜ / simulationstate.parameters.κ))))
        end

        order.price = maximum([0, simulationstate.LOB.bₜ + 1 + η]) # ensure that there are no offers with negative prices
        order.volume = round(Int, PowerLaw(5, α))
        order.displayVolume = order.volume
    else
        α = 1 + (simulationstate.LOB.ρₜ/simulationstate.parameters.ν)

        # if spread is 0 set η = 0
        if simulationstate.LOB.sₜ == 0
            η = 0
        else
            η = floor(rand(Gamma(simulationstate.LOB.sₜ, exp(-simulationstate.LOB.ρₜ / simulationstate.parameters.κ))))
        end

        order.price = maximum([0, simulationstate.LOB.aₜ - 1 -  η]) # ensure that there are no bids with negative prices
        order.volume = round(Int, PowerLaw(5, α))
        order.displayVolume = order.volume
    end

    # only record event times if they are after the initializing of the LOB
    if !(simulationstate.initializing)
        SubmitOrder(simulationstate.gateway, order)
        current_time = Dates.now()
        push!(highfrquency.actionTimes, current_time - simulationstate.start_time)
        push!(highfrquency.currentOrders, (current_time, order))
        simulationstate.event_counter += 1
    else
        # if initializing do not allow agents to submit an order in the spread
        if (order.price > simulationstate.LOB.aₜ) || (order.price < simulationstate.LOB.bₜ)
            SubmitOrder(simulationstate.gateway, order)
            #current_time = Dates.now()
            #push!(highfrquency.currentOrders, (current_time, order))
            simulationstate.event_counter += 1
        end
    end
    
end
function ChartistAction(chartist::Chartist, simulationstate::SimulationState)
    # TODO: 

    # if the order book is being initialized do nothing
    if simulationstate.initializing 
        return
    end

    if !(Dates.now() - simulationstate.start_time < simulationstate.parameters.T)
        return
    end

    order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("TF", chartist.traderId), type = "Market")

    # Update the agent's EWMA
    chartist.p̄ₜ += chartist.λ * (simulationstate.LOB.mₜ - chartist.p̄ₜ) # took away the lambda

    ######## Set agent's actions

    # boolean saying whether there are orders on the contra side (assume there isn't)
    contra = false

    # boolean saying if the order will cause a volatility auction (assume it won't)
    volatility = false

    if chartist.p̄ₜ > simulationstate.LOB.mₜ + (1/2) * simulationstate.LOB.sₜ  
        order.side = "Sell"
    elseif chartist.p̄ₜ < simulationstate.LOB.mₜ - (1/2) * simulationstate.LOB.sₜ 
        order.side = "Buy"
    else # if there this agent is not trading then return
        return
    end

    # set the order parameters
    xₘ = 20
    if abs(simulationstate.LOB.mₜ - chartist.p̄ₜ) > (simulationstate.parameters.δ * simulationstate.LOB.mₜ)
        xₘ = 50
    end
    α = order.side == "Sell" ? 1 - (simulationstate.LOB.ρₜ/simulationstate.parameters.ν) : 1 + (simulationstate.LOB.ρₜ/simulationstate.parameters.ν)
	if (order.side == "Buy" && !isempty(simulationstate.LOB.asks)) || (order.side == "Sell" && !isempty(simulationstate.LOB.bids)) # Agent won't submit MO if no orders on contra side
		order.volume = round(Int, PowerLaw(xₘ, α))
        contra = true
	end
    if order.side == "Sell" # Agent won't send MO if it will cause a volatility auction
        if (abs(simulationstate.LOB.priceReference - simulationstate.LOB.bₜ) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    else
        if (abs(simulationstate.LOB.aₜ - simulationstate.LOB.priceReference) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    end

    # Update τ and order times before the trade

    # submit order if there are orders on the other side and if the order won't cause a volatility auction
    if (contra) && !(volatility)
        SubmitOrder(simulationstate.gateway, order)
        push!(chartist.actionTimes, Dates.now() - simulationstate.start_time)
        simulationstate.event_counter += 1
    else # if there are no contra orders or a volatility auction might happen return
        return
    end
end

function FundamentalistAction(fundamentalist::Fundamentalist, simulationstate::SimulationState)
    # TODO: (1) Check the generation of the mid-price

    # if (event_counter > max_events + number_initial_messages)
    #     return
    # end

    # if the order book is being initialized do nothing
    if simulationstate.initializing 
        return
    end
    
    if !(Dates.now() - simulationstate.start_time < simulationstate.parameters.T)
        return
    end

    order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("VI", fundamentalist.traderId), type = "Market")

    ######## Set agent's actions

    # boolean saying whether there are orders on the contra side (assume there isn't)
    contra = false

    # boolean saying if the order will cause a volatility auction (assume it won't)
    volatility = false

    if fundamentalist.fₜ < simulationstate.LOB.mₜ - (1/2) * simulationstate.LOB.sₜ  
        order.side = "Sell"
    elseif fundamentalist.fₜ > simulationstate.LOB.mₜ + (1/2) * simulationstate.LOB.sₜ 
        order.side = "Buy"
    else # if this agent is not trading then return
        return
    end

    # set the parameters of the order
    xₘ = 20
    if abs(simulationstate.LOB.mₜ - fundamentalist.fₜ) > (simulationstate.parameters.δ * simulationstate.LOB.mₜ)
        xₘ = 50
    end
    # order.side = fundamentalist.fₜ < LOB.mₜ ? "Sell" : "Buy" # NEED TO CHANGE
    α = order.side == "Sell" ? 1 - (simulationstate.LOB.ρₜ/simulationstate.parameters.ν) : 1 + (simulationstate.LOB.ρₜ/simulationstate.parameters.ν)
	if (order.side == "Buy" && !isempty(simulationstate.LOB.asks)) || (order.side == "Sell" && !isempty(simulationstate.LOB.bids))
        order.volume = round(Int, PowerLaw(xₘ, α))
        contra = true
	end
    if order.side == "Sell" # Agent won't send MO if it will cause a volatility auction
        if (abs(simulationstate.LOB.priceReference - simulationstate.LOB.bₜ) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    else
        if (abs(simulationstate.LOB.aₜ - simulationstate.LOB.priceReference) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    end

    # submit order if there are orders on the other side and if the order won't cause a volatility auction
    if (contra) && !(volatility)
        SubmitOrder(simulationstate.gateway, order)
        push!(fundamentalist.actionTimes, Dates.now() - simulationstate.start_time)
        simulationstate.event_counter += 1
    else
        return
    end
    
end

# Define the trading for the RL agent and the updating of the Q-matrix
function RLAction(rlAgent::RL, simulationstate::SimulationState)

    # if the order book is being initialized do nothing
    if simulationstate.initializing 
        return
    end
    current_time = Dates.now()
    if !(current_time - simulationstate.start_time < simulationstate.parameters.T) # might also need to change if multiple sessions (assumes end of RL is always before end of sim)
        return
    end

    # if the agent has traded all the volume before the simulation is up don't trade 
    # println()
    # println()
    # println("Done = ", rlAgent.done)
    if rlAgent.done
        # println("Inventory Remaining = ", rlAgent.i)
        return
    end

    # process the RL messages to get traded volume (for inventory counter) and reward = Σpᵢvᵢ 
    total_volume_traded, sum_price_volume, trade_message = ProcessMessages(simulationstate.rlMessages, rlAgent)

    if trade_message != ""
        push!(rlAgent.trade_messages, trade_message)
    end

    # get the remaining volume (need the RL messages)
    rlAgent.i = rlAgent.i - total_volume_traded

    # println("Volume Traded = ", total_volume_traded)
    # println("Inventory Remaining = ", rlAgent.i)
    # println("Σpᵢvᵢ = ", sum_price_volume)

    # get remaining time (will need to change if we use more than 1 execution per day), also need to add that entire reamining volume must be traded
    rl_start_time = (simulationstate.start_time + simulationstate.rlParameters.startTime)
    rl_end_time = (rl_start_time + simulationstate.rlParameters.T)
    remaining_time = (rl_end_time - current_time).value
    # println("Remaining Time = ", remaining_time)

    # get the new state
    state, done = GetState(simulationstate.LOB, remaining_time, rlAgent.i, simulationstate.rlParameters, rlAgent)

    # tells the rl agent whether there is inventory to trade or not
    rlAgent.done = done

    # set the reward
    reward = sum_price_volume

    # # add the store the rewards 
    if rlAgent.prev_action >= 0 
        push!(rlAgent.R, reward) 
    end

    # if the prev state said time is up but there is still remaining volume then penalize the agent for remaining volume, set terminal state and done
    if rlAgent.prev_action >= 0
        if rlAgent.prev_state[1] == 0 && !(rlAgent.done)
            rlAgent.done = true
            state = [0,0,0,0]
        end
    end

    # println("State = ", state)
    # println("Terminal = ", rlAgent.done)

    # update Q only if an action has been taken in the sim (so after the first iteration of the event loop)
    # note argmax for selling and argmin for buying agents
    if rlAgent.prev_action >= 0
        # rlAgent.Q[rlAgent.prev_state][rlAgent.prev_action] += 1
        rlAgent.Q[rlAgent.prev_state][rlAgent.prev_action] = rlAgent.Q[rlAgent.prev_state][rlAgent.prev_action] + simulationstate.rlParameters.α * (reward + simulationstate.rlParameters.discount_factor * maximum(rlAgent.Q[state]) - rlAgent.Q[rlAgent.prev_state][rlAgent.prev_action])
    end

    # find the action that minimizes (maximizes) the cost (profit) based on the current state
    policy_probabilities = EpisilonGreedyPolicy(rlAgent.Q, state, simulationstate.rlParameters.ϵ, rlAgent)
    action = sample(Xoshiro(), 1:simulationstate.rlParameters.A, Weights(policy_probabilities), 1)[1] # supply a different rng than the global one so for the same price path different actions can be generated
    action_percentage_volume = simulationstate.rlParameters.actions[action]
    # action = rand([1, 2])
    # action_percentage_volume = simulationstate.rlParameters.actions[action]

    # add the store the rewards 
    if rlAgent.prev_action >= 0 
        push!(rlAgent.actions, action) 
    end

    # check remaining time and if non-left trade everything and not in terminal state
    if remaining_time <= 0
        if !(rlAgent.done)
            # println("State = ", state, " | Prev State = ", rlAgent.prev_state)
            action = simulationstate.rlParameters.A
            action_percentage_volume = 2 
        elseif (rlAgent.done) # if time is up and we are in terminal then dont trade
            action = 1
            action_percentage_volume = 0
        end
    end

    # perform the action
    order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("RL", rlAgent.traderId), type = "Market")
    order.side = rlAgent.actionType

    # boolean saying whether there are orders on the contra side (assume there isn't)
    contra = false

    # boolean saying if the order will cause a volatility auction (assume it won't)
    volatility = false

    # set volume and add contra and volatility check (don't let RL agent trade if no order on the other side or if it will cause a volatility auction)
    # action_percentage_volume 
    if (order.side == "Buy" && !isempty(simulationstate.LOB.asks)) || (order.side == "Sell" && !isempty(simulationstate.LOB.bids))
        order.volume = minimum([ceil(action_percentage_volume * simulationstate.rlParameters.Ntwap), rlAgent.i]) # adjust twap volume action_percentage_volume, trade remaining volume if action is too high
        contra = true
	end
    if order.side == "Sell" # Agent won't send MO if it will cause a volatility auction
        if (abs(simulationstate.LOB.priceReference - simulationstate.LOB.bₜ) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    else
        if (abs(simulationstate.LOB.aₜ - simulationstate.LOB.priceReference) / simulationstate.LOB.priceReference) > 0.1
            order.volume = 0
            volatility = true
        end
    end

    # submit order if there are orders on the other side and if the order won't cause a volatility auction and order.volume must be greater than 0
    if (contra) && !(volatility) && (order.volume > 0)
        SubmitOrder(simulationstate.gateway, order)
        push!(rlAgent.actionTimes, Dates.now() - simulationstate.start_time)
        simulationstate.event_counter += 1        
    end

    # update next state and action to reflect the state and action combination taken in this iteration 
    rlAgent.prev_state = state
    rlAgent.prev_action = action

    # println("Action = ", action)
    # println("Action Percentage Volume = ", action_percentage_volume)
    # println("Action Volume = ", order.volume)
    # println("Contra = ", contra)
    # println("Volatility = ", volatility)

    return

end

#---------------------------------------------------------------------------------------------------

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

# RL
Rocket.on_next!(actor::RL, simulationstate::SimulationState) = RLAction(actor, simulationstate)
Rocket.on_error!(actor::RL, err)      = error(err)
Rocket.on_complete!(actor::RL)        = println("ID: " * actor.traderId * " Name: " * actor.traderMnemonic * " Completed!")

#---------------------------------------------------------------------------------------------------

#----- Define Subject -----#

Rocket.on_next!(subject::Subject, simulationstate::SimulationState) = nextState(subject, simulationstate)

#---------------------------------------------------------------------------------------------------

#----- Update LOB state -----#
function UpdateLOBState!(LOB::LOBState, message)
    # TODO: (1) 

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
        if type == :Trade && fields[3][1:2] == "HF" && side == :Buy 
            # checks asks for a buy since this will be where the LO buys is executed against
            if !(id in keys(LOB.asks)) 
                # this is the excess order that needs to be a new limit order
                type = Symbol("New")
                
                # if it is the first then I need to remove some volume
                if i == 1
                    volume = volume - sum(parse(Int, split(e, ",")[3]) for e in executions[2:end])
                end

            end
        elseif type == :Trade && fields[3][1:2] == "HF" && side == :Sell 
            # checks bids for a sell since this will be where the LO buys is executed against
            if !(id in keys(LOB.bids))
                # this is the excess order that needs to be a new limit order
                type = Symbol("New")

                # if it is the first then I need to remove some volume
                if i == 1
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
#---------------------------------------------------------------------------------------------------

#----- Initialize LOB -----#
function InitializeLOB(simulationstate::SimulationState, messages_chnl::Channel, source::Subject, number_initial_messages::Int64, initial_messages_received::Vector{String}, messages_received::Vector{String})
    # TODO (1) Write initial orders to a file

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
                UpdateLOBState!(simulationstate.LOB, message)

            end 

        elseif isready(messages_chnl) && simulationstate.event_counter > number_initial_messages

            while isready(messages_chnl)
                message = take!(messages_chnl)

                # push to the initial messages array
                push!(initial_messages_received, message)

                # use the messages_recieved array to keep track of the number of messages in the channel
                popfirst!(messages_received)

                # update the LOBState with new order 
                UpdateLOBState!(simulationstate.LOB, message)

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

###################### Tommorow
#                      (1) 
function simulate(parameters::Parameters, rlParameters::RLParameters, gateway::TradingGateway, rlTraders::Bool, rlTraining::Bool, print_and_plot::Bool, write_messages::Bool, write_volume_spread::Bool; seed::Int64, iteration::Int64)
    # TODO: Refactor  

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
    simulationstate = SimulationState(LOB, parameters, rlParameters, Vector{String}(), gateway, true, 1, Dates.now())

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
    rl_traders_vec = nothing
    if rlTraders
        # make a copy of the initial Q (change when there are multiple agents)
        Q = DefaultDict{Vector{Int64}, Vector{Float64}}(() -> zeros(Float64, rlParameters.A))
        for key in keys(rlParameters.initialQ)
            Q[key] = copy(rlParameters.initialQ[key])
        end
        rl_traders_vec = map(i -> RL(i, "RL"*string(i), Array{Millisecond,1}(), Vector{String}(), Vector{Int64}(), rlParameters.type, rlParameters.T.value, rlParameters.V, false, Vector{Float64}(), Q, Vector{Int64}(), -1.0), 1:rlParameters.Nᵣₗ) 
    end

    # initialize the Subject and subscribe actors to it
    source = Subject(SimulationState)
    map(i -> subscribe!(source, i), hf_traders_vec)
    map(i -> subscribe!(source, i), char_traders_vec)
    map(i -> subscribe!(source, i), fun_traders_vec)
    if rlTraders
        map(i -> subscribe!(source, i), rl_traders_vec)
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
    # println("\n#################################################################### Initialization Started\n")

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

    # println("\n#################################################################### Initialization Done\n")

    #----- Event Loop -----#

    # currently only 1 message from the channel can be processed in a
    # single loop takes about 0.006 seconds (maybe even less)

    # get the current time to use for while loop (causes the first HF trade to be a bit before the LF traders but after it seems fine)
    simulationstate.start_time = Dates.now()

    #Dates.now() - current_time <= parameters.T
    try
        @time while true
            
            # Sleep the main task for a tiny amount of time to switch to the listening task
            sleep(0.05)

            # send the event to be processed by the actors
            if isready(messages_chnl) # if there are messages in the channel then take the last update

                # update the LOB with all the new events
                while isready(messages_chnl)

                    message = take!(messages_chnl)
                    
                    # update the LOBState with new order 
                    UpdateLOBState!(simulationstate.LOB, message)

                    # if the message was from an RL trader then add this to the RL messages Vector
                    if rlTraders

                        # take the time out of the message
                        msg = split(message, "|")[2:end]

                        #msg = split(message, "|")
                        fields = split(msg[1], ",")

                        # need 2 types (Type is the original one and type is the one that changes (crossing LOs))
                        trader = fields[3]

                        if occursin("RL", trader)
                            push!(simulationstate.rlMessages, message)
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

                # clear the RL messages vector (only want to store current orders)
                empty!(simulationstate.rlMessages)

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
        return nothing, nothing
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
        # println()
    #     for key in keys(rl_traders_vec[1].Q)
    #         println(key, " => ", rl_traders_vec[1].Q[key], " | Total actions = ", sum(rl_traders_vec[1].Q[key]))
    #     end
    #     println()

    println()
    #     println("Rewards = ", rl_traders_vec[1].R)
    #     println("Total Reward = ", sum(rl_traders_vec[1].R))
        println("Number of Actions = ", length(rl_traders_vec[1].R))
    # println("Trade Messages = ", rl_traders_vec[1].trade_messages)
    #     println("Number of Trades = ", length(rl_traders_vec[1].trade_messages))
    println()
    end

    # return mid-prices and micro-prices, and if rl traded then return the Q matrices and rewards for the simulations
    if rlTraders # extend for multiple rl agents
        rl_result = Dict("Q" => rl_traders_vec[1].Q, "Rewards" => rl_traders_vec[1].R, "TotalReward" => sum(rl_traders_vec[1].R), "Actions" => rl_traders_vec[1].actions, "NumberActions" => length(rl_traders_vec[1].actions), "NumberTrades" => length(rl_traders_vec[1].trade_messages))
        return mid_prices, micro_prices, rl_result
    else
        return mid_prices, micro_prices
    end


end
