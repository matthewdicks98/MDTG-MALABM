#=
ReactiveABM:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie, (some code was adapted from https://github.com/IvanJericevich/IJPCTG-ABMCoinTossX)
- Function: Functions used to simulate the event based ABM with a maximum of 1 RL selling agent
- Structure: 
    
- Example:
    
- Prerequisites:

=#
ENV["JULIA_COPY_STACKS"]=1

using Dates, DataStructures

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"

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

# defines the type 1 RL agent (REFACTOR THIS AND TYPE 2 SO THAT COMMON FIELDS ARE STORED IN ONE STRUCT TYPE)
mutable struct RL1 <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    agentType::String      # defines that it is an RL agent type 1
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a decision
    actions::Vector{Int64}            # store all the actions the agent took
    actionType::String     # defines if the agent will be buying or selling
    activated::Bool         # defines if the agent has been activated and will have taken an action before
    t::Int64               # time remaining
    i::Int64               # inventory remaining/getting             
    done::Bool             # indicates if the trader has traded all the volume
    R::Vector{Float64}   # rewards stored over the course of a single execution
    Q::DefaultDict       # is the Q matrix for the RL agent for a single execution (only add a state-action pair when it is seen for the first time, don't initialize full Q)
    prev_state::Vector{Int64}     # stores the previous state to determine where we are in sim
    currentMOs::OrderedDict{Int64,Dict{String,Any}} # stores the current market orders in play
    trade_vwap::Float64           # stores the vwap for the agents stratergy
    total_trade_volume::Int64   # stores total volume traded
    total_price_volume::Int64   # stores Σpᵢvᵢ
end

# defines the type 2 RL agent
mutable struct RL2 <: Actor{SimulationState}
    traderId::Int64 # uniquely identifies an agent
    traderMnemonic::String # used to record who sent the orders
    agentType::String      # defines that it is an RL agent type 1
    actionTimes::Array{Millisecond,1} # Arrival process of when each agent makes a decision
    actions::Vector{Int64}            # store all the actions the agent took
    actionType::String     # defines if the agent will be buying or selling
    activated::Bool         # defines if the agent has been activated and will have taken an action before
    t::Int64               # time remaining
    i::Int64               # inventory remaining/getting             
    done::Bool             # indicates if the trader has traded all the volume
    R::Vector{Float64}   # rewards stored over the course of a single execution
    Q::DefaultDict       # is the Q matrix for the RL agent for a single execution (only add a state-action pair when it is seen for the first time, don't initialize full Q)
    prev_state::Vector{Int64}     # stores the previous state to determine where we are in sim

    currentMOs::OrderedDict{Int64,Dict{String,Any}} # stores the current market orders in play

    currentLOs::OrderedDict{Int64,Dict{String,Any}} # stores the current LOs, {LO_ID => {state => , action => , LimitOrder => }}
    cancelledOrders::Set{Int64} # set of order IDs storing the order that have been cancelled (used to determine why an order is not in LOB)

    trade_vwap::Float64           # stores the vwap for the agents stratergy
    total_trade_volume::Int64   # stores total volume traded
    total_price_volume::Int64   # stores Σpᵢvᵢ
end

#---------------------------------------------------------------------------------------------------

#----- Agent rules -----# 
function HighFrequencyAgentAction(highfrquency::HighFrequency, simulationstate::SimulationState)

    # do not trade if trading time is finished
    if !(simulationstate.initializing)
        if !(Dates.now() - simulationstate.start_time < simulationstate.parameters.T)
            return
        end
    end

    # cancel orders that have not been matched (dont have to account for the order being in the book or not since an empty message is sent back from CTX)
    if !(simulationstate.initializing) && length(highfrquency.currentOrders) > 0 # dont cancell orders during initialization and make sure there are orders to cancel
        # check if oldest order needs to be cancelled
        current_time = Dates.now()
        if current_time - highfrquency.currentOrders[1][1] > simulationstate.parameters.γ

            timed_out_inds = Vector{Int64}()
            for (i, (t, o)) in enumerate(highfrquency.currentOrders)
                if (current_time - t) <= simulationstate.parameters.γ
                    break
                else
                    push!(timed_out_inds, i)
                end
            end

            # get all the orders that are in the LOB
            cancel_inds = Vector{Int64}()
            for ind in timed_out_inds
                if (highfrquency.currentOrders[ind][2].orderId in keys(simulationstate.LOB.bids)) || (highfrquency.currentOrders[ind][2].orderId in keys(simulationstate.LOB.asks))
                    push!(cancel_inds, ind)
                end
            end

            # send cancellation orders through for order still in LOB
            for ind in cancel_inds
                CancelOrder(simulationstate.gateway, highfrquency.currentOrders[ind][2])
            end
            
            # delete the timed out orders from the currentOrders array
            deleteat!(highfrquency.currentOrders, timed_out_inds) 

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
        if ((order.price > simulationstate.LOB.aₜ) || (order.price < simulationstate.LOB.bₜ))
            SubmitOrder(simulationstate.gateway, order)
            simulationstate.event_counter += 1
        end
    end

    return
    
end
function ChartistAction(chartist::Chartist, simulationstate::SimulationState)

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
function RLAction1(rlAgent::RL1, simulationstate::SimulationState)

    # if the order book is being initialized do nothing
    if simulationstate.initializing 
        return
    end
    current_time = Dates.now()
    if !(current_time - simulationstate.start_time < simulationstate.parameters.T) 
        return
    end
    if rlAgent.done
        return
    end

    # process the RL messages (MO only) to get traded volume (for the prev state and prev action combination) (for inventory counter) and reward = Σpᵢvᵢ 
    # for this agent type we only need the market orders and these are 1 per event loop
    matched_order_id, executed_volume, executed_price_volume, submitted_state, submitted_action = (-1, 0, 0, [], -1)
    market_order_results = ProcessMarketOrders(rlAgent)
    if length(market_order_results) > 0
        matched_order_id, executed_volume, executed_price_volume, submitted_state, submitted_action = market_order_results[1]
    end

    # get the remaining volume (need the RL messages)
    rlAgent.i = rlAgent.i - executed_volume

    # get remaining time (will need to change if we use more than 1 execution per day), also need to add that entire reamining volume must be traded
    rl_start_time = (simulationstate.start_time + simulationstate.rlParameters.startTime)
    rl_end_time = (rl_start_time + simulationstate.rlParameters.T)
    remaining_time = (rl_end_time - current_time).value
    past_time = ((current_time - rl_start_time).value) / 1000 # convert to seconds for better reward function

    # get the new state
    state, done = GetState(simulationstate.LOB, remaining_time, rlAgent.i, simulationstate.rlParameters, rlAgent)

    # tells the rl agent whether there is inventory to trade or not
    rlAgent.done = done

    # update VWAP price for current rl agent trades (only update if there has been a trade with more than 0 volume executed)
    if executed_volume > 0
        rlAgent.trade_vwap = (1 / (rlAgent.total_trade_volume + executed_volume)) * (rlAgent.total_price_volume + executed_price_volume)
        rlAgent.total_trade_volume += executed_volume
        rlAgent.total_price_volume += executed_price_volume # Σpᵢvᵢ
    end

    # compute the market VWAP (do not include trades of the current agent)
    market_vwap_volume = simulationstate.total_trade_volume
    market_vwap_price_volume = simulationstate.total_price_volume
    if length(simulationstate.rl_traders_vec) > 1 # there are more than 1 RL trader add their contribution to the VWAP market price
        market_vwap_volume += sum(rl_trader.total_trade_volume for rl_trader in simulationstate.rl_traders_vec if rl_trader.traderId != rlAgent.traderId)
        market_vwap_price_volume += sum(rl_trader.total_price_volume for rl_trader in simulationstate.rl_traders_vec if rl_trader.traderId != rlAgent.traderId)
    end

    # check if market vwap is not 0 if it is then set it to initial mid-price
    if market_vwap_volume > 0
        market_vwap = market_vwap_price_volume / market_vwap_volume
    else
        market_vwap = simulationstate.parameters.m₀
    end

    # Always make sure that the a < minimum volume allowed by the RL agent actions
    reward = 0
    executed_volume == 0 ? a = 1 : a = executed_volume
    rlAgent.i == 0 ? penalty = 0 : penalty = (simulationstate.rlParameters.λᵣ * exp(simulationstate.rlParameters.γᵣ * past_time) * (rlAgent.i/a))
    if rlAgent.actionType == "Sell"  # only recompute the reward if vwaps are not 0
        rlAgent.total_trade_volume > 0 ? reward = (log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4 - penalty : reward = 0 - log(market_vwap) # only compute the reward if there has been volume traded
    elseif rlAgent.actionType == "Buy"
        rlAgent.total_trade_volume > 0 ? reward = -(log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4 - penalty : reward = 0 - log(market_vwap) # penalty same as above (change)
    end

    println()
    println("RL agent: ", rlAgent.traderMnemonic)
    println("Agent type: ", rlAgent.agentType)
    println("Action type: ", rlAgent.actionType)
    println("Activated: ", rlAgent.activated)
    println("Remaining time: ", remaining_time)
    println("Current state: ", state)
    println("Previous state: ", rlAgent.prev_state)
    println("Submitted state: ", submitted_state)
    println("Submitted action: ", submitted_action)
    println("Total volume traded: ", rlAgent.total_trade_volume)
    println("Price * volume: ", rlAgent.total_price_volume)
    println("Action volume: ", executed_volume)
    println("Action price * volume: ", executed_price_volume)
    println("ABM agents volume: ", simulationstate.total_trade_volume)
    println("ABM agents price * volume: ", simulationstate.total_price_volume)
    println("Market vwap: ", market_vwap)
    println("Agent vwap: ", rlAgent.trade_vwap)
    println("Penalty: ", penalty)
    if rlAgent.actionType == "Sell"
        println("Slippage: ", (log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4)
    elseif rlAgent.actionType == "Buy"
        println("Slippage: ", -(log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4)
    end
    println("Reward: ", reward)
    println()

    # if the prev state said time is up but there is still remaining volume then penalize the agent for remaining volume, set terminal state and done
    if rlAgent.activated
        if rlAgent.prev_state[1] == 0 && !(rlAgent.done)
            rlAgent.done = true
            state = [0,0,0,0]
        end

        # add the store the rewards
        push!(rlAgent.R, reward) 

        # update Q only if an action has been taken in the sim (so after the first iteration of the event loop)
        rlAgent.Q[submitted_state][submitted_action] = rlAgent.Q[submitted_state][submitted_action] + simulationstate.rlParameters.α * (reward + simulationstate.rlParameters.discount_factor * maximum(rlAgent.Q[state]) - rlAgent.Q[submitted_state][submitted_action])

    end

    # find the action that minimizes (maximizes) the cost (profit) based on the current state
    policy_probabilities = EpisilonGreedyPolicy(rlAgent.Q, state, simulationstate.rlParameters.ϵ)
    action = sample(Xoshiro(), 1:simulationstate.rlParameters.A1, Weights(policy_probabilities), 1)[1] # supply a different rng than the global one so for the same price path different actions can be generated
    action_percentage_volume = simulationstate.rlParameters.actionsRL1[action]

    # check remaining time and if non left  and not in terminal state trade as much as possible
    if remaining_time <= 0
        if !(rlAgent.done)
            action = simulationstate.rlParameters.A1
            action_percentage_volume = simulationstate.rlParameters.actionsRL1[action] 
        elseif (rlAgent.done) # if time is up and we are in terminal then dont trade
            action = 1
            action_percentage_volume = simulationstate.rlParameters.actionsRL1[action]
        end
    end

    # add the store the actions 
    if rlAgent.activated 
        push!(rlAgent.actions, action) 
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

    # store the selected action
    push!(rlAgent.currentMOs, simulationstate.event_counter => Dict("state" => state, "action" => action, "order" => order, "time" => Dates.now(), "trade_message" => ""))
    simulationstate.event_counter += 1        

    # submit order if there are orders on the other side and if the order won't cause a volatility auction and order.volume must be greater than 0
    if (contra) && !(volatility) && (order.volume > 0)
        SubmitOrder(simulationstate.gateway, order)
        push!(rlAgent.actionTimes, Dates.now() - simulationstate.start_time)
    end

    # update next state and action to reflect the state and action combination taken in this iteration 
    rlAgent.prev_state = state

    # after first being called activate the agent so updating and storing can start next iteration
    if !(rlAgent.activated)
        rlAgent.activated = true
    end

    return

end

# Define the trading for the RL agent and the updating of the Q-matrix
function RLAction2(rlAgent::RL2, simulationstate::SimulationState)

    # if the order book is being initialized do nothing
    if simulationstate.initializing 
        return
    end
    current_time = Dates.now()
    if !(current_time - simulationstate.start_time < simulationstate.parameters.T) 
        return
    end
    if rlAgent.done
        return
    end

    ####### Cancell timed out orders
    if length(collect(rlAgent.currentLOs)) > 0 # make sure there are orders to cancel
        # check if oldest order needs to be cancelled
        pairs_list = collect(rlAgent.currentLOs)
        if current_time - pairs_list[1][2]["time"] > simulationstate.parameters.γ

            # find all orders that need to be cancelled
            timed_out_inds = Vector{Int64}()
            for (i, key) in enumerate(keys(rlAgent.currentLOs))
                if (current_time - rlAgent.currentLOs[key]["time"]) <= simulationstate.parameters.γ
                    break
                else
                    push!(timed_out_inds, i)
                end
            end

            # get all the orders that are in the LOB (dont removed cancelled orders currentLOs, will be done by ProcessLimitOrders)
            cancel_inds = Vector{Int64}()
            for ind in timed_out_inds
                if (pairs_list[ind][1] in keys(simulationstate.LOB.bids)) || (pairs_list[ind][1] in keys(simulationstate.LOB.asks))
                    push!(cancel_inds, ind)
                end
            end

            # send cancellation orders through for orders still in LOB store cancelled orders
            for ind in cancel_inds
                CancelOrder(simulationstate.gateway, pairs_list[ind][2]["order"])
                push!(rlAgent.cancelledOrders, pairs_list[ind][1])
            end

        end
    end

    # process the orders based on their updated states
    market_order_results = ProcessMarketOrders(rlAgent)
    limit_order_results = ProcessLimitOrders(simulationstate, rlAgent)
    all_order_results = [limit_order_results; market_order_results] # limit orders will always accur before the market order

    println()
    println("Market order resutls: ",market_order_results)
    println("Limit order resutls: ",limit_order_results)
    println("All order results: ", all_order_results)
    println()

    println()
    println("Current market orders: ", rlAgent.currentMOs)
    println("Current limit orders: ", rlAgent.currentLOs)
    println()

    ####### Compute state information

    # get the remaining volume using all messages to get current state
    total_executed_volume = 0
    if length(all_order_results) > 0
        total_executed_volume = sum(executed_volume for (_, executed_volume, _, _, _) in all_order_results)
    end
    rlAgent.i = rlAgent.i - total_executed_volume

    # get remaining time (will need to change if we use more than 1 execution per day), also need to add that entire reamining volume must be traded
    rl_start_time = (simulationstate.start_time + simulationstate.rlParameters.startTime)
    rl_end_time = (rl_start_time + simulationstate.rlParameters.T)
    remaining_time = (rl_end_time - current_time).value
    past_time = ((current_time - rl_start_time).value) / 1000 # convert to seconds for better reward function

    # # get the new state
    state, done = GetState(simulationstate.LOB, remaining_time, rlAgent.i, simulationstate.rlParameters, rlAgent)

    # tells the rl agent whether there is inventory to trade or not
    rlAgent.done = done

    ####### Update VWAPs, compute the reward, and update

    # compute the market VWAP (do not include trades of the current agent)
    market_vwap_volume = simulationstate.total_trade_volume
    market_vwap_price_volume = simulationstate.total_price_volume
    if length(simulationstate.rl_traders_vec) > 1 # there are more than 1 RL trader add their contribution to the VWAP market price
        market_vwap_volume += sum(rl_trader.total_trade_volume for rl_trader in simulationstate.rl_traders_vec if rl_trader.traderId != rlAgent.traderId)
        market_vwap_price_volume += sum(rl_trader.total_price_volume for rl_trader in simulationstate.rl_traders_vec if rl_trader.traderId != rlAgent.traderId)
    end

    # check if market vwap is not 0 if it is then set it to initial mid-price
    if market_vwap_volume > 0
        market_vwap = market_vwap_price_volume / market_vwap_volume
    else
        market_vwap = simulationstate.parameters.m₀
    end

    for (orderId, executed_volume, executed_price_volume, submitted_state, submitted_action) in all_order_results

        if executed_volume > 0
            rlAgent.trade_vwap = (1 / (rlAgent.total_trade_volume + executed_volume)) * (rlAgent.total_price_volume + executed_price_volume)
            rlAgent.total_trade_volume += executed_volume
            rlAgent.total_price_volume += executed_price_volume # Σpᵢvᵢ
        end

        # Always make sure that the a < minimum volume allowed by the RL agent actions
        reward = 0
        executed_volume == 0 ? a = 1 : a = executed_volume
        rlAgent.i == 0 ? penalty = 0 : penalty = (simulationstate.rlParameters.λᵣ * exp(simulationstate.rlParameters.γᵣ * past_time) * (rlAgent.i/a))
        if rlAgent.actionType == "Sell"  # only recompute the reward if vwaps are not 0
            rlAgent.total_trade_volume > 0 ? reward = (log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4 - penalty : reward = 0 - log(market_vwap) # only compute the reward if there has been volume traded
        elseif rlAgent.actionType == "Buy"
            rlAgent.total_trade_volume > 0 ? reward = -(log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4 - penalty : reward = 0 - log(market_vwap) # penalty same as above (change)
        end

        println()
        println("RL agent: ", rlAgent.traderMnemonic)
        println("Agent type: ", rlAgent.agentType)
        println("Action type: ", rlAgent.actionType)
        println("Activated: ", rlAgent.activated)
        println("Done: ", rlAgent.done)
        println("Remaining time: ", remaining_time)
        println("Remaining inventory: ", rlAgent.i)
        println("Current state: ", state)
        println("Previous state: ", rlAgent.prev_state)
        println("Submitted state: ", submitted_state)
        println("Submitted action: ", submitted_action)
        println("Total volume traded: ", rlAgent.total_trade_volume)
        println("Price * volume: ", rlAgent.total_price_volume)
        println("Action volume: ", executed_volume)
        println("Action price * volume: ", executed_price_volume)
        println("ABM agents volume: ", simulationstate.total_trade_volume)
        println("ABM agents price * volume: ", simulationstate.total_price_volume)
        println("Market vwap: ", market_vwap)
        println("Agent vwap: ", rlAgent.trade_vwap)
        println("Penalty: ", penalty)
        if rlAgent.actionType == "Sell"
            println("Slippage: ", (log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4)
        elseif rlAgent.actionType == "Buy"
            println("Slippage: ", -(log(rlAgent.trade_vwap) - log(market_vwap)) * 1e4)
        end
        println("Reward: ", reward)
        println()

        # if the prev state said time is up but there is still remaining volume then penalize the agent for remaining volume, set terminal state and done
        if rlAgent.activated

            # add the store the rewards
            push!(rlAgent.R, reward) 

            # update Q only if an action has been taken in the sim (so after the first iteration of the event loop)
            rlAgent.Q[submitted_state][submitted_action] = rlAgent.Q[submitted_state][submitted_action] + simulationstate.rlParameters.α * (reward + simulationstate.rlParameters.discount_factor * maximum(rlAgent.Q[state]) - rlAgent.Q[submitted_state][submitted_action])

        end
    end
    ############################# LOOP ENDS HERE

    if rlAgent.activated
        if rlAgent.prev_state[1] == 0 && !(rlAgent.done)
            rlAgent.done = true
            state = [0,0,0,0]
        end
    end

    # find the action that minimizes (maximizes) the cost (profit) based on the current state
    policy_probabilities = EpisilonGreedyPolicy(rlAgent.Q, state, simulationstate.rlParameters.ϵ)
    action = sample(Xoshiro(), 1:simulationstate.rlParameters.A2, Weights(policy_probabilities), 1)[1] # supply a different rng than the global one so for the same price path different actions can be generated
    action_percentage_volume, delta = simulationstate.rlParameters.actionsRL2[action]

    # check remaining time and if non left and not in terminal state trade as much as possible (largest market order)
    if remaining_time <= 0
        if !(rlAgent.done)
            action = findlast(x -> x == -1, last.(collect(values(simulationstate.rlParameters.actionsRL2))))
            action_percentage_volume, delta = simulationstate.rlParameters.actionsRL2[action] 
        elseif (rlAgent.done) # if time is up and we are in terminal then dont trade
            action = 1
            action_percentage_volume, delta = simulationstate.rlParameters.actionsRL2[action]
        end
    end

    # store the actions 
    if rlAgent.activated 
        push!(rlAgent.actions, action) 
    end

    println()
    println("RL agent: ", rlAgent.traderMnemonic)
    println("Agent type: ", rlAgent.agentType)
    println("Action type: ", rlAgent.actionType)
    println("Activated: ", rlAgent.activated)
    println("Done: ", rlAgent.done)
    println("Remaining time: ", remaining_time)
    println("Remaining inventory: ", rlAgent.i)
    println("Current state: ", state)
    println("Previous state: ", rlAgent.prev_state)
    println("Action: ", action)
    println()

    ####### Submit the order based on action selected

    # perform the action
    if delta == -1 # market order
        order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("RL", rlAgent.traderId), type = "Market")
        order.side = rlAgent.actionType
    else
        order = Order(orderId = simulationstate.event_counter, traderMnemonic = string("RL", rlAgent.traderId), type = "Limit")
        order.side = rlAgent.actionType
    end
    
    # boolean saying whether there are orders on the contra side (assume there isn't)
    contra = false

    # boolean saying if the order will cause a volatility auction (assume it won't)
    volatility = false

    # compute volume in LOB, used to make sure we never submit orde where we can have negative inventory
    lo_book_volume = 0
    for lo_order_id in keys(rlAgent.currentLOs)
        lo_book_volume += rlAgent.currentLOs[lo_order_id]["order"].volume
    end

    if order.type == "Market"
        # set volume and add contra and volatility check (don't let RL agent trade if no order on the other side or if it will cause a volatility auction)
        if (order.side == "Buy" && !isempty(simulationstate.LOB.asks)) || (order.side == "Sell" && !isempty(simulationstate.LOB.bids))
            order.volume = max(0, minimum([ceil(action_percentage_volume * simulationstate.rlParameters.Ntwap), (rlAgent.i - lo_book_volume)])) # adjust twap volume action_percentage_volume, trade remaining volume if action is too high
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

        push!(rlAgent.currentMOs, simulationstate.event_counter => Dict("state" => state, "action" => action, "order" => order, "time" => Dates.now(), "trade_message" => ""))
        simulationstate.event_counter += 1

        if (contra) && !(volatility) && (order.volume > 0)
            SubmitOrder(simulationstate.gateway, order)
            push!(rlAgent.actionTimes, Dates.now() - simulationstate.start_time)
        end

    elseif order.type == "Limit"
        order.volume = max(0, minimum([ceil(action_percentage_volume * simulationstate.rlParameters.Ntwap), (rlAgent.i - lo_book_volume)]))
        if order.side == "Buy"
            order.price = simulationstate.LOB.bₜ - delta
            order.displayVolume = order.volume
        elseif order.side == "Sell"
            order.price = simulationstate.LOB.aₜ + delta
            order.displayVolume = order.volume
        end
        push!(rlAgent.currentLOs, simulationstate.event_counter => Dict("state" => state, "action" => action, "order" => order, "time" => Dates.now(), "matched_volume" => 0, "status" => "live"))
        simulationstate.event_counter += 1   

        if order.volume > 0
            SubmitOrder(simulationstate.gateway, order)
            push!(rlAgent.actionTimes, Dates.now() - simulationstate.start_time)
        end
    end

    println()
    println("Remaining Inventory: ", rlAgent.i)
    println("LO book volume: ", lo_book_volume)
    println("Max allowed: ", (rlAgent.i - lo_book_volume))
    println("action_percentage_volume * simulationstate.rlParameters.Ntwap: ", 2 * simulationstate.rlParameters.Ntwap)
    println("Order volume: ", order.volume)
    println()

    ####### Store info about order just submitted

    # update next state and action to reflect the state and action combination taken in this iteration 
    rlAgent.prev_state = state

    # after first being called activate the agent so updating and storing can start next iteration
    if !(rlAgent.activated)
        rlAgent.activated = true
    end

    return

end


#---------------------------------------------------------------------------------------------------