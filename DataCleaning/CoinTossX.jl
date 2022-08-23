#=
DataCleaning:
- Julia version: 1.7.1
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie, (some edits and bug fixes Matthew Dicks)
- Function: Clean CoinTossX simulated data into L1LOB data for stylized fact analysis as well as visualisation of HFT time-series
- Structure:
    1. Supplementary functions
    2. Clean raw data into L1LOB format
    3. Plot simulation results
- Examples:
    CleanData("Raw", initialization = false)
    VisualiseSimulation("TAQ", "L1LOB")
=#
using CSV, DataFrames, Dates, Plots, Plots.PlotMeasures, Tables, ProgressMeter

# set working directory (the path to the DataCleaning/CoinTossX.jl file)
path_to_folder = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/DataCleaning"
cd(path_to_folder)

#----- Supplementary functions -----#
mutable struct Order
    Price::Int64
    Volume::Int64
end
mutable struct Best
    Price::Int64
    Volume::Int64
    IDs::Vector{Int64}
end
function MidPrice(best::Best, contraBest::Best)
    return (isempty(best.IDs) || isempty(contraBest.IDs)) ? missing : (best.Price + contraBest.Price) / 2
end
function MicroPrice(best::Best, contraBest::Best)
    return (isempty(best.IDs) || isempty(contraBest.IDs)) ? missing : (best.Price * best.Volume + contraBest.Price * contraBest.Volume) / (best.Volume + contraBest.Volume)
end
function Spread(best::Best, contraBest::Best)
    return (isempty(best.IDs) || isempty(contraBest.IDs)) ? missing : abs(best.Price - contraBest.Price)
end
function OrderImbalance(bids::Dict{Int64, Order}, asks::Dict{Int64, Order})
    if isempty(bids) && isempty(asks)
        return missing
    elseif isempty(bids)
        return -1
    elseif isempty(asks)
        return 1
    else
        totalBuyVolume = sum(order.Volume for order in values(bids))
        totalSellVolume = sum(order.Volume for order in values(asks))
        return (totalBuyVolume - totalSellVolume) / (totalBuyVolume + totalSellVolume)
    end
end
function DepthProfile(lob::Dict{Int64, Order}, side::Int64)
    profile = zeros(Union{Int64, Missing}, 7)
    prices = map(x -> x.Price, values(lob)) |> unique |> x -> side == 1 ? sort(x, rev = true) : sort(x, rev = false)
    for p in 1:7
        if p <= length(prices)
            profile[p] = sum(v.Volume for v in values(lob) if v.Price == prices[p])
        else
            profile[p] = missing
        end
    end
    return profile
end
#---------------------------------------------------------------------------------------------------

#----- Clean raw data into L1LOB format -----#
#=
Function:
    - Update LOB and best with LO
    - Full crossed orders are also added to the LOB and then aggressed with subsequent effective MOs
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - side = ∈ {-1, 1}
    - allowCrossing = should crossed orders be handled or not
Output:
    - Order id of crossed order (if any)
    - All other variables are updated in-place
=#
function ProcessLimitOrder!(file::IOStream, order::DataFrameRow, best::Best, contraBest::Best, lob::Dict{Int64, Order}, side::Int64)
    if isempty(lob) || isempty(best.IDs) # If the dictionary is empty, this order automatically becomes best
        best.Price = order.Price; best.Volume = order.Volume; best.IDs = [order.OrderId]
        midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
        println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", side, ",", midPrice, ",", microPrice, ",", spread))
    else # Otherwise find the best
        if (side * order.Price) > (side * best.Price) # Change best if price of current order better than the best (side == 1 => order.Price > best.Price) (side == -1 => order.Price < best.Price)
            best.Price = order.Price; best.Volume = order.Volume; best.IDs = [order.OrderId] # New best is created
            if !isempty(contraBest.IDs) && (side * order.Price) > (side * contraBest.Price) # Crossing order
                error("Negative spread at order " * string(order.OrderId))
            else # Only print the LO if it is not a CLO
                midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
                println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", side, ",", midPrice, ",", microPrice, ",", spread))
            end
        elseif order.Price == best.Price # Add the new order's volume and orderid to the best if they have the same price
            best.Volume += order.Volume; push!(best.IDs, order.OrderId) # Best is ammended by adding volume to best and appending the order id
            midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
            println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", side, ",", midPrice, ",", microPrice, ",", spread))
        end
    end
    push!(lob, order.OrderId => Order(order.Price, order.Volume)) # New order is always pushed to LOB dictionary only after best is processed
end
#=
Function:
    - Update LOB and best with MO
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - side = ∈ {-1, 1}
Output:
    - All variables are updated in-place
=#
function ProcessMarketOrder!(file::IOStream, order::DataFrameRow, nextOrder::Symbol, best::Best, contraBest::Best, lob::Dict{Int64, Order}, side::Int64)
    contraOrder = lob[order.OrderId] # Extract order on contra side
    if order.Volume == best.Volume # Trade filled best - remove from LOB, and update best
        delete!(lob, order.OrderId) # Remove the order from the LOB
        if !isempty(lob) # If the LOB is non empty find the best
            bestPrice = side * maximum(x -> side * x.Price, values(lob)) # Find the new best price (bid => side == 1 so find max price) (ask => side == -1 so find min price)
            indeces = [k for (k, v) in lob if v.Price == bestPrice] # Find the order ids of the best
            best.Price = bestPrice; best.Volume = sum(lob[i].Volume for i in indeces); best.IDs = indeces # Update the best
        else # If the LOB is empty remove best
            best.Price = 0; best.Volume = 0; best.IDs = Vector{Int64}()
        end
    else # Trade partially filled best
        if order.Volume == contraOrder.Volume # Trade filled contra order - remove order from LOB, remove order from best, and update best
            delete!(lob, order.OrderId)
            best.Volume -= order.Volume; best.IDs = setdiff(best.IDs, order.OrderId)
        else # Trade partially filled contra order - update LOB, update best
            lob[order.OrderId].Volume -= order.Volume
            best.Volume -= order.Volume
        end
    end
    if order.Trader != nextOrder
        midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
        !isempty(best.IDs) ? println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", side, ",", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.Initialization) * "," * string(order.DateTime, ",missing,missing,Limit,", side, ",missing,missing,missing"))
    end
end
#=
Function:
    - Update LOB and best with OC
    - Append mid-price, micro-price and spread info
Arguments:
    - file = output file to which L1LOB will be printed
    - order = order to be processed
    - best = best bid (ask) if bid (ask) LO
    - contraBest = best ask (bid) if ask (bid) LO
    - lob = bid (ask) side of LOB if bid (ask) LO
    - side = ∈ {-1, 1}
Output:
    - All variables are updated in-place
=#
function ProcessCancelOrder!(file::IOStream, order::DataFrameRow, best::Best, contraBest::Best, lob::Dict{Int64, Order}, side::Int64)
    delete!(lob, order.OrderId) # Remove the order from the LOB
    if order.OrderId in best.IDs # Cancel hit the best
        if !isempty(lob) # Orders still remain in the LOB - find and update best
            bestPrice = side * maximum(x -> side * x.Price, values(lob)) # Find the new best price (bid => side == 1 so find max price) (ask => side == -1 so find min price)
            indeces = [k for (k,v) in lob if v.Price == bestPrice] # Find the order ids of the best
            best.Price = bestPrice; best.Volume = sum(lob[i].Volume for i in indeces); best.IDs = indeces # Update the best
            midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
            println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Cancelled,", side, ",", midPrice, ",", microPrice, ",", spread))
        else # The buy side LOB was emptied - update best
            best.Price = 0; best.Volume = 0; best.IDs = Vector{Int64}()
            println(file, string(order.Initialization) * "," * string(order.DateTime, ",missing,missing,Cancelled,", side, ",missing,missing,missing"))
        end
    end # OC did not hit best
end
#=
Function:
    - Process all orders and clean raw TAQ data into L1LOB bloomberg format
Arguments:
    - orders = TAQ data
    - allowCrossing = should crossed orders be handled or not
Output:
    - TAQ data
    - Output L1LOB file written to csv
=#
function CleanData(raw::String; initialization::Bool = false, times::Vector{Millisecond} = Vector{Millisecond}())
    println("Reading in data...")
    orders = CSV.File(string("../Data/CoinTossX/Raw.csv"), drop = [!isempty(times) ? :DateTime : :Nothing], types = Dict(:Initialization => Symbol, :ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol, :Type => Symbol, :TraderMnemonic => Symbol), dateformat = "yyyy-mm-ddTHH:MM:SS.s") |> DataFrame
    replace!(orders.Type, :New => :Limit); # Rename Types # orders.Type[findall(x -> x == 0, orders.Price)] .= :Market 
    orders.ClientOrderId[findall(x -> x == :Cancelled, orders.Type)] .*= -1
    DataFrames.rename!(orders, [:ClientOrderId => :OrderId, :TraderMnemonic => :Trader])
    if !isempty(times) orders.DateTime = zeros(Float64, nrow(orders)) end; orders.Imbalance = zeros(Union{Missing, Float64}, nrow(orders)) # Calculate the order imbalance in the LOB
    bids = Dict{Int64, Order}(); asks = Dict{Int64, Order}() # Both sides of the entire LOB are tracked with keys corresponding to orderIds
    bestBid = Best(0, 0, Vector{Int64}()); bestAsk = Best(0, 0, Vector{Int64}()) # Current best bid/ask is stored in a tuple (Price, vector of Volumes, vector of OrderIds) and tracked
    bidDepthProfile = zeros(Union{Int64, Missing}, nrow(orders), 7); askDepthProfile = zeros(Union{Int64, Missing}, nrow(orders), 7)
    LO_ask_delay = Vector{DataFrameRow}()
    LO_bid_delay = Vector{DataFrameRow}()
    open("../Data/CoinTossX/L1LOB.csv", "w") do file
        println(file, "Initialization,DateTime,Price,Volume,Type,Side,MidPrice,MicroPrice,Spread") # Header
        @showprogress "Cleaning Data..." for i in 1:nrow(orders) # Iterate through all orders
            order = orders[i, :]
            #-- Limit Orders --#
            if order.Type == :Limit
                if !isempty(times) order.DateTime = Dates.value(pop!(times)) / 1000 end
                if order.Side == :Buy # Buy limit order
                    ProcessLimitOrder!(file, order, bestBid, bestAsk, bids, 1) # Add the order to the lob and update the best if necessary
                else # Sell limit order
                    ProcessLimitOrder!(file, order, bestAsk, bestBid, asks, -1) # Add the order to the lob and update the best if necessary
                end
            #-- Market Orders --#
            elseif order.Type == :Trade # Market order always affects the best
                if i == nrow(orders) # if last order is a market order then break due to cases in dealing with MO
                    break
                end
                if !isempty(times)
                    if orders[i + 1, :Type] != :Trade
                        order.DateTime = Dates.value(pop!(times)) / 1000 # Last trade
                    else
                        order.DateTime = Dates.value(times[end]) / 1000 # Not the last trade so don't remove time
                    end
                end
                if order.Side == :Sell # Trade was buyer-initiated (Sell MO)
                    ## Deal with LOs that crossed the spread
                    # LO that crossed the spread, if id not in bids and it is the last trade for the agent, the last field is the new LO
                    # check if last trade in trades, check if traders are equal or if trades are equal then the times must be diff
                    if (orders[i, :Trader] != orders[i + 1, :Trader] && !(order.OrderId in keys(bids))) || (orders[i, :Trader] == orders[i + 1, :Trader] && orders[i, :DateTime] != orders[i+1, :DateTime] && !(order.OrderId in keys(bids)))
                        # Combined sell trade is printed after the last split trade with VWAP price
                        indeces = (findprev(x -> x != orders[i, :Trader], orders.Trader, i) + 1):(i-1)
                        println(file, string(order.Initialization) * "," * string(order.DateTime, ",", round(Int, sum(orders[indeces, :Price] .* orders[indeces, :Volume]) / sum(orders[indeces, :Volume])), ",", sum(orders[indeces, :Volume]), ",Market,-1,missing,missing,missing"))
                        # print the updated bests bid due to the trade
                        best = bestBid
                        contraBest = bestAsk
                        midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
                        !isempty(best.IDs) ? println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", 1, ",", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.Initialization) * "," * string(order.DateTime, ",missing,missing,Limit,", 1, ",missing,missing,missing"))
                        # process the LO due to the excess
                        order.Type = :Limit
                        ProcessLimitOrder!(file, order, bestAsk, bestBid, asks, -1) # Add the order to the lob and update the best if necessary
                    # LO that crossed the spread, if id not in bids and it is the first trade for the agent, the last field is the new LO with reduced volume
                    elseif !(order.OrderId in keys(bids))
                        order.Type = :Limit
                        indeces = (i+1):(findnext(x -> x != orders[i, :Trader], orders.Trader, i) - 1)
                        order.Volume -= sum(orders[indeces, :Volume])
                        if order.Volume > 0
                            #ProcessLimitOrder!(file, order, bestAsk, bestBid, asks, -1) # Add the order to the lob and update the best if necessary
                            push!(LO_ask_delay, order)
                        end
                    # regular trade or a LO that crossed the spread without clearing one side of the LOB
                    elseif (orders[i, :Trader] != orders[i + 1, :Trader]) || (orders[i, :Trader] == orders[i + 1, :Trader] && orders[i, :DateTime] != orders[i+1, :DateTime]) 
                        # find trades just made by a given agent
                        indeces = (findprev(x -> x != orders[i, :Trader], orders.Trader, i) + 1):i
                        agent_trades = orders[indeces,:]
                        # get only the ones that are market orders
                        indeces = findall(x -> x == :Trade, agent_trades.Type)
                        println(file, string(order.Initialization) * "," * string(order.DateTime, ",", round(Int, sum(agent_trades[indeces, :Price] .* agent_trades[indeces, :Volume]) / sum(agent_trades[indeces, :Volume])), ",", sum(agent_trades[indeces, :Volume]), ",Market,-1,missing,missing,missing")) # Combined sell trade is printed after the last split trade with VWAP price
                        ProcessMarketOrder!(file, order, orders[i + 1, :Trader], bestBid, bestAsk, bids, 1) # Sell trade affects bid side. Always aggress MO against contra side and update LOB and best
                        if !isempty(LO_ask_delay)
                            order_delay = popfirst!(LO_ask_delay) # it will only ever have 1 order in it
                            ProcessLimitOrder!(file, order_delay, bestAsk, bestBid, asks, -1)
                        end
                    else
                        ProcessMarketOrder!(file, order, orders[i + 1, :Trader], bestBid, bestAsk, bids, 1) # Sell trade affects bid side. Always aggress MO against contra side and update LOB and best
                    end
                    
                else # Trade was seller-initiated (Buy MO)
                    ## Deal with LOs that crossed the spread
                    # LO that crossed the spread, if id not in asks and it is the last trade for the agent, the last field is the new LO
                    if orders[i, :Trader] != orders[i + 1, :Trader] && !(order.OrderId in keys(asks)) || (orders[i, :Trader] == orders[i + 1, :Trader] && orders[i, :DateTime] != orders[i+1, :DateTime] && !(order.OrderId in keys(asks)))
                        # Combined sell trade is printed after the last split trade with VWAP price
                        indeces = (findprev(x -> x != orders[i, :Trader], orders.Trader, i) + 1):(i-1)
                        println(file, string(order.Initialization) * "," * string(order.DateTime, ",", round(Int, sum(orders[indeces, :Price] .* orders[indeces, :Volume]) / sum(orders[indeces, :Volume])), ",", sum(orders[indeces, :Volume]), ",Market,1,missing,missing,missing"))                        
                        # print the updated bests bid due to the trade
                        best = bestAsk
                        contraBest = bestBid
                        midPrice = MidPrice(best, contraBest); microPrice = MicroPrice(best, contraBest); spread = Spread(best, contraBest)
                        !isempty(best.IDs) ? println(file, string(order.Initialization) * "," * string(order.DateTime, ",", best.Price, ",", best.Volume, ",Limit,", -1, ",", midPrice, ",", microPrice, ",", spread)) : println(file, string(order.Initialization) * "," * string(order.DateTime, ",missing,missing,Limit,", -1, ",missing,missing,missing"))
                        # process the excess LO
                        order.Type = :Limit
                        ProcessLimitOrder!(file, order, bestBid, bestAsk, bids, 1) # Add the order to the lob and update the best if necessary
                    # LO that crossed the spread, if id not in asks and it is the first trade for the agent, the last field is the new LO with reduced volume
                    elseif !(order.OrderId in keys(asks)) # can just check if id is not in because the only other option has been checked above
                        order.Type = :Limit
                        indeces = (i+1):(findnext(x -> x != orders[i, :Trader], orders.Trader, i) - 1)
                        order.Volume -= sum(orders[indeces, :Volume])
                        if order.Volume > 0
                            #ProcessLimitOrder!(file, order, bestBid, bestAsk, bids, 1) # Add the order to the lob and update the best if necessary
                            push!(LO_bid_delay, order)
                        end
                    # regular trade or a LO that crossed the spread without clearing one side of the LOB
                    elseif (orders[i, :Trader] != orders[i + 1, :Trader]) || (orders[i, :Trader] == orders[i + 1, :Trader] && orders[i, :DateTime] != orders[i+1, :DateTime])
                        # find trades just made by a given agent
                        indeces = (findprev(x -> x != orders[i, :Trader], orders.Trader, i) + 1):i
                        agent_trades = orders[indeces,:]
                        # get only the ones that are market orders
                        indeces = findall(x -> x == :Trade, agent_trades.Type)
                        println(file, string(order.Initialization) * "," * string(order.DateTime, ",", round(Int, sum(agent_trades[indeces, :Price] .* agent_trades[indeces, :Volume]) / sum(agent_trades[indeces, :Volume])), ",", sum(agent_trades[indeces, :Volume]), ",Market,1,missing,missing,missing")) # Combined buy trade is printed after the last split trade with VWAP price
                        ProcessMarketOrder!(file, order, orders[i + 1, :Trader], bestAsk, bestBid, asks, -1) # buy trade affects ask side. Always aggress MO against contra side and update LOB and best
                        if !isempty(LO_bid_delay)
                            order_delay = popfirst!(LO_bid_delay)
                            ProcessLimitOrder!(file, order_delay, bestBid, bestAsk, bids, 1)
                        end
                    else
                        ProcessMarketOrder!(file, order, orders[i + 1, :Trader], bestAsk, bestBid, asks, -1) # buy trade affects ask side. Always aggress MO against contra side and update LOB and best
                    end
                end

            #-- Cancel Orders --#
            elseif order.Type == :Cancelled
                if !isempty(times) order.DateTime = Dates.value(pop!(times)) / 1000 end
                if order.Side == :Buy # Cancel buy limit order
                    ProcessCancelOrder!(file, order, bestBid, bestAsk, bids, 1) # Aggress cancel order against buy side and update LOB and best
                else # Cancel sell limit order
                    ProcessCancelOrder!(file, order, bestAsk, bestBid, asks, -1) # Aggress cancel order against sell side and update LOB and best
                end
            else
                if !isempty(times) order.DateTime = Dates.value(times[end]) / 1000 end # Trades appearing hereafter require the same timestamp
            end
            orders[i, :Imbalance] = OrderImbalance(bids, asks) # Calculate the volume imbalance after every iteration
            bidDepthProfile[i, :] = DepthProfile(bids, 1); askDepthProfile[i, :] = DepthProfile(asks, -1)
        end
    end
    CSV.write("../Data/CoinTossX/TAQ.csv", orders)
    CSV.write("../Data/CoinTossX/DepthProfileData.csv",  Tables.table(hcat(bidDepthProfile, askDepthProfile)), writeheader=false)
end
#---------------------------------------------------------------------------------------------------

#----- Plot simulation results -----#
function VisualiseSimulation(taq::String, l1lob::String; format = "pdf", endTime = missing, startTime = missing)
    # Cleaning
    orders = CSV.File(string("../Data/CoinTossX/", taq, ".csv"), missingstring = "missing", types = Dict(:Trader => Symbol, :Initialization => Symbol, :Side => Symbol, :Type => Symbol, :DateTime => DateTime)) |> DataFrame |> x -> filter(y -> y.Type != :Trade, x) |> x -> filter(y -> y.Initialization != :INITIAL, x)
    l1lob = CSV.File(string("../Data/CoinTossX/", l1lob, ".csv"), missingstring = "missing", types = Dict(:Initialization => Symbol, :Type => Symbol, :DateTime => DateTime)) |> DataFrame |> x -> filter(y -> x.Type != :Market, x) |> x -> filter(y -> y.Initialization != :INITIAL, x)# Filter out trades from L1LOB since their mid-prices are missing
    if !ismissing(startTime)
        filter!(x -> x.DateTime >= startTime, l1lob); filter!(x -> x.DateTime >= startTime, orders)
    end
    if !ismissing(endTime)
        filter!(x -> x.DateTime <= endTime, l1lob); filter!(x -> x.DateTime <= endTime, orders)
    end
    orders.RelativeTime = @. Dates.value(orders.DateTime - orders.DateTime[1]) / 1000
    l1lob.RelativeTime = @. Dates.value(l1lob.DateTime - l1lob.DateTime[1]) / 1000
    asks = filter(x -> x.Type == :Limit && x.Side == :Sell, orders); bids = filter(x -> x.Type == :Limit && x.Side == :Buy, orders)
    sells = filter(x -> x.Type == :Market && x.Side == -1, l1lob); buys = filter(x -> x.Type == :Market && x.Side == 1, l1lob)
    cancelAsks = filter(x -> x.Type == :Cancelled && x.Side == :Sell, orders); cancelBids = filter(x -> x.Type == :Cancelled && x.Side == :Buy, orders)
    # Bubble plot
    bubblePlot = plot(asks.RelativeTime, asks.Price, seriestype = :scatter, marker = (:red, stroke(:red), 0.5), label = "Ask (LO)", ylabel = "Price (ticks)", legend = :outertopright, legendfontsize = 5, tickfontsize = 5, xticks = false, fg_legend = :transparent)
    plot!(bubblePlot, bids.RelativeTime, bids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), 0.5), label = "Bid (LO)")
    plot!(bubblePlot, sells.RelativeTime, sells.Price, seriestype = :scatter, marker = (:red, stroke(:red), :dtriangle, 0.5), label = "Sell (MO)")
    plot!(bubblePlot, buys.RelativeTime, buys.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :utriangle, 0.5), label = "Buy (MO)")
    plot!(bubblePlot, cancelAsks.RelativeTime, cancelAsks.Price, seriestype = :scatter, marker = (:red, stroke(:red), :xcross, 0.5), label = "Cancel Ask")
    plot!(bubblePlot, cancelBids.RelativeTime, cancelBids.Price, seriestype = :scatter, marker = (:blue, stroke(:blue), :xcross, 0.5), label = "Cancel Bid")
    # L1LOB features
    plot!(bubblePlot, l1lob.RelativeTime, l1lob.MicroPrice, seriestype = :line, linecolor = :green, label = "Micro-price")
    plot!(bubblePlot, l1lob.RelativeTime, l1lob.MidPrice, seriestype = :steppost, linecolor = :black, label = "Mid-price", linewidth = 2)
    # Spread and imbalance features
    imbalance = filter(x -> x != "",orders.Imbalance)
    if typeof(imbalance[1]) == String31 || typeof(imbalance[1]) == String # imbalance can constain strings
        imbalance = parse.(Float64, imbalance)
    end
    volumeImbalance = plot(orders.RelativeTime, imbalance, seriestype = :line, linecolor = :purple, xlabel = "Time (s)", ylabel = "Order Imbalance", label = "OrderImbalance", legend = :topleft, legendfontsize = 5, tickfontsize = 5, xrotation = 30, fg_legend = :transparent, right_margin = 8mm)
    plot!(twinx(), l1lob.RelativeTime, l1lob.Spread, seriestype = :steppost, linecolor = :orange, ylabel = "Spread", label = "Spread", legend = :topright, legendfontsize = 5, tickfontsize = 5, xaxis = false, xticks = false, fg_legend = :transparent)
    # Log-return plot
    filter!(x -> !ismissing(x.MidPrice), l1lob)
    logreturns = diff(log.(l1lob.MidPrice))
    returns = plot(l1lob.RelativeTime[2:end], logreturns, seriestype = :line, linecolor = :black, legend = false, tickfontsize = 5, ylabel = "Log-returns", xticks = false)
    l = @layout([a; b{0.2h}; c{0.2h}])
    simulation = plot(bubblePlot, returns, volumeImbalance, layout = l, link = :x, guidefontsize = 7)
    savefig(simulation, "../Images/CoinTossX/Simulation." * format)
    println("Simuation visualization complete")
end
#---------------------------------------------------------------------------------------------------

# CleanData("Raw", initialization = false)
# VisualiseSimulation("TAQ", "L1LOB")
