#=
JSE:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie
- Function: Clean raw JSE L1LOB TAQ data
- Structure:
    1. Create the DataFrame
    2. Trade and quote compacting
    3. Compute the mid price, mid price change, micro price, trade inter arrivals, normalized volume
    4. Classify trades (Lee/Ready)
- Example:
    makeCleanTaqData("NPN")
=#
using CSV, DataFrames, Dates, ProgressMeter, Plots, LaTeXStrings, TimeSeries, Tables

# set working directory (the path to the Scripts/StylisedFacts.jl file)
path_to_folder = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/DataCleaning"
cd(path_to_folder)

#----- Create the dataframe that holds all the data -----#
function createDataFrame(data::DataFrame)

    # get all the trading dates (pulls dates out of timestamp)
    dates = Date.(data[:,1])

    # get all the unique dates (all the days)
    days = unique(dates)

    # create the dataframe that I am going to populate
    full_df = DataFrame(timeStamp = DateTime[], date = Date[], time = Time[], eventType = String[], bid = Float64[], bidVol = Float64[],
    ask = Float64[], askVol = Float64[], trade = Float64[], tradeVol = Float64[], normTradeVol = Float64[], midPrice = Float64[],
    midPriceChange = Float64[], microPrice = Float64[], interArrivals = Float64[], tradeSign = String[])

    # for each day compute values for each event
    @showprogress "Creating full_df..." for j in 1:length(days)

        # get data for a single day
        days_date = days[j]
        days_data = data[findall(x -> x == days_date, dates), :]

        # filter for continuous trading data 9:00 - 16:49:59
        start = DateTime(days_date) + Hour(9)
        close = DateTime(days_date) + Hour(16) + Minute(50)
        days_cont_data = days_data[findall(x -> start <= x && x < close, days_data[:,:times]),:]

        # loop through all the events in a day and create the dataframe with all the ask, bid and trade data
        for i in 1:size(days_cont_data)[1]
            # get event
            event = days_cont_data[i,:]
            # populate dates and times
            timeStamp_i = event[:times]
            date_i = Date.(event[:times])
            time_i = Time.(event[:times])
            # populate the event columns
            if event[:type] == "BID"
                line = (timeStamp_i, date_i, time_i, "BID", event[:value], event[:size], NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, "")
                push!(full_df, line)
            elseif event[:type] == "ASK"
                line = (timeStamp_i, date_i, time_i, "ASK", NaN, NaN, event[:value], event[:size], NaN, NaN, NaN, NaN, NaN, NaN, NaN, "")
                push!(full_df, line)
            elseif event[:type] == "TRADE"
                # only keep AT (automated trading) trades
                if event[:condcode] == "AT"
                    line = (timeStamp_i, date_i, time_i, "TRADE", NaN, NaN, NaN, NaN, event[:value], event[:size], NaN, NaN, NaN, NaN, NaN, "")
                    push!(full_df, line)
                end
            end
        end
    end

    # add an id column to help identify rows (might have to do this to speed things up if we get more data)
    full_df[!, :id] = 1:size(full_df)[1]

    return full_df

end
#---------------------------------------------------------------------------------------------------

#----- Trade and quote compacting -----#
function compact(full_df::DataFrame)

    # get all the unique time stamps
    time_stamps = unique(full_df[:,:timeStamp])

    # create the df that will hold the compacted data
    full_compact_df = DataFrame(timeStamp = DateTime[], date = Date[], time = Time[], eventType = String[], bid = Float64[], bidVol = Float64[],
    ask = Float64[], askVol = Float64[], trade = Float64[], tradeVol = Float64[], normTradeVol = Float64[], midPrice = Float64[],
    midPriceChange = Float64[], microPrice = Float64[], interArrivals = Float64[], tradeSign = String[], id = Int64[])

    @showprogress "Compacting..." for i in 1:length(time_stamps)

        # find out if the timestamp had a trade
        # get the current time stamp
        current_time_stamp = time_stamps[i]

        # get all the events for a given day that occured at a current time stamp
        time_stamp_events = full_df[findall(x -> x == current_time_stamp, full_df[:,:timeStamp]),:]

        # get all the trades in the given time stamp
        all_trade_inds = findall(x -> x == "TRADE", time_stamp_events[:,:eventType])

        # if trades did occur in the timestamp
        if all_trade_inds != []
            # get the trades
            trades = time_stamp_events[all_trade_inds, :]
            # get the trade prices and volumes
            trade_prices = trades[:,:trade]
            trade_vols = trades[:,:tradeVol]
            # compute the volume weighted average trade price
            vwap_price = sum(trade_prices .* trade_vols)/sum(trade_vols)
            # compute the comulative volume
            cummulative_vol = sum(trade_vols)
            # push to the DataFrame (using 1 for id as a place holder)
            temp_trade = (current_time_stamp, Date.(current_time_stamp), Time.(current_time_stamp), "TRADE", NaN, NaN, NaN, NaN, vwap_price, cummulative_vol, NaN, NaN, NaN, NaN, NaN, "",1)
            # determine if there were quotes before the trade
            first_trade_ind = findfirst(x -> x == "TRADE", time_stamp_events[:,:eventType])
            # if the first trade was not the first thing that happened in that time step then add the last ask and bid before that
            if first_trade_ind > 1
                # get all events before the trade occurred
                events_before_first_trade = time_stamp_events[1:(first_trade_ind - 1),:]
                # get the last Bid in the given time stamp (quote compacting is done by fetching the most recent quote)
                best_bid_ind = findlast(x -> x == "BID", events_before_first_trade[:,:eventType])
                # get the last ask in the given time stamp (quote compacting is done by fetching the most recent quote)
                best_ask_ind = findlast(x -> x == "ASK", events_before_first_trade[:,:eventType])
                if !isnothing(best_bid_ind) && !isnothing(best_ask_ind)
                    # maintain order of the quotes
                    if best_bid_ind < best_ask_ind
                        # push bid then ask
                        best_bid_before = events_before_first_trade[best_bid_ind,:]
                        push!(full_compact_df, best_bid_before)
                        best_ask_before = events_before_first_trade[best_ask_ind,:]
                        push!(full_compact_df, best_ask_before)
                    elseif best_ask_ind < best_bid_ind
                        # push ask before bid
                        best_ask_before = events_before_first_trade[best_ask_ind,:]
                        push!(full_compact_df, best_ask_before)
                        best_bid_before = events_before_first_trade[best_bid_ind,:]
                        push!(full_compact_df, best_bid_before)
                    end
                elseif !isnothing(best_bid_ind)
                    # no ask so just push bid
                    best_bid_before = events_before_first_trade[best_bid_ind,:]
                    push!(full_compact_df, best_bid_before)
                elseif !isnothing(best_ask_ind)
                    # no bid so just push bid
                    best_ask_before = events_before_first_trade[best_ask_ind,:]
                    push!(full_compact_df, best_ask_before)
                end
            end
            # add trade to the dataframe
            push!(full_compact_df, temp_trade)
        end
        # always push the last ask and last bid of the timestamp

        # get the last Bid in the given time stamp (quote compacting is done by fetching the most recent quote)
        best_bid_ind = findlast(x -> x == "BID", time_stamp_events[:,:eventType])

        # get the last ask in the given time stamp (quote compacting is done by fetching the most recent quote)
        best_ask_ind = findlast(x -> x == "ASK", time_stamp_events[:,:eventType])

        if !isnothing(best_bid_ind) && !isnothing(best_ask_ind)
            # maintain order of the quotes
            if best_bid_ind < best_ask_ind
                # push bid then ask
                best_bid_before = time_stamp_events[best_bid_ind,:]
                push!(full_compact_df, best_bid_before)
                best_ask_before = time_stamp_events[best_ask_ind,:]
                push!(full_compact_df, best_ask_before)
            elseif best_ask_ind < best_bid_ind
                # push ask before bid
                best_ask_before = time_stamp_events[best_ask_ind,:]
                push!(full_compact_df, best_ask_before)
                best_bid_before = time_stamp_events[best_bid_ind,:]
                push!(full_compact_df, best_bid_before)
            end
        elseif !isnothing(best_bid_ind)
            # no ask so just push bid
            best_bid_before = time_stamp_events[best_bid_ind,:]
            push!(full_compact_df, best_bid_before)
        elseif !isnothing(best_ask_ind)
            # no bid so just push bid
            best_ask_before = time_stamp_events[best_ask_ind,:]
            push!(full_compact_df, best_ask_before)
        end
    end

    # we have nulled out the ids so need to create a new id column
    full_compact_df[:,:id] = 1:size(full_compact_df)[1]

    return full_compact_df

end
#---------------------------------------------------------------------------------------------------

#----- Compute the mid price, mid price change, micro price, trade inter arrivals, normalized volume -----#
function computeMidMicroArrivalsNormVol(full_df::DataFrame)

    # get all the trading dates (get the dates from datetime object)
    dates = Date.(full_df[:,1])

    # get all the unique dates (all the days)
    days = unique(dates)

    # count the current row we are in, in the dataframe
    row_ind = 1

    @showprogress "Computing full_df..." for j in 1:length(days)

        # get data for a single day
        days_date = days[j]
        days_cont_data = full_df[findall(x -> x == days_date, dates), :]

        # loop through all the events in a day, compute the computable columns of full_df
        for i in 1:size(days_cont_data)[1]
            # get event
            event = days_cont_data[i,:]
            # now compute the mid price
            if event[:eventType] == "BID"
                # get the current best bid info
                best_bid = event[:bid]
                best_bid_vol = event[:bidVol]
                # get the best ask price up until this event
                best_ask_ind = findlast(x -> x == "ASK", days_cont_data[1:i,:eventType])
                # there are no best ask offers (eg begining of day)
                if isnothing(best_ask_ind)
                    full_df[row_ind, :midPrice] = NaN
                    full_df[row_ind, :microPrice] = NaN
                else # there is a current best ask
                    # get the best ask data
                    best_ask = days_cont_data[best_ask_ind, :ask]
                    best_ask_vol = days_cont_data[best_ask_ind, :askVol]
                    # compute the mid and micro price
                    full_df[row_ind, :midPrice] = 0.5 * (best_bid + best_ask)
                    full_df[row_ind, :microPrice] = (best_ask_vol / (best_ask_vol + best_bid_vol)) * best_ask + (best_bid_vol / (best_ask_vol + best_bid_vol)) * best_bid
                end
            elseif event[:eventType] == "ASK"
                # get the current best ask info
                best_ask = event[:ask]
                best_ask_vol = event[:askVol]
                # get the best bid price up until this event
                best_bid_ind = findlast(x -> x == "BID", days_cont_data[1:i,:eventType])
                # there are no best bid offers
                if isnothing(best_bid_ind)
                    full_df[row_ind, :midPrice] = NaN
                    full_df[row_ind, :microPrice] = NaN
                else # there is a current best ask
                    # get the best ask data
                    best_bid = days_cont_data[best_bid_ind, :bid]
                    best_bid_vol = days_cont_data[best_bid_ind, :bidVol]
                    # compute the mid and micro price
                    full_df[row_ind, :midPrice] = 0.5 * (best_bid + best_ask)
                    full_df[row_ind, :microPrice] = (best_ask_vol / (best_ask_vol + best_bid_vol)) * best_ask + (best_bid_vol / (best_ask_vol + best_bid_vol)) * best_bid
                end
            elseif event[:eventType] == "TRADE"
                # mid price does not change until after trade (new best ask or bid) but during it is still the same
                # (if trade is first thing that heppens in the day then NaN)
                # i = 1, no bid or ask, i = 2 one side of the order book is empty
                if i > 2
                    full_df[row_ind, :midPrice] = full_df[row_ind-1, :midPrice]
                    full_df[row_ind, :microPrice] = full_df[row_ind-1, :microPrice]
                else
                    full_df[row_ind, :midPrice] = NaN
                    full_df[row_ind, :microPrice] = NaN
                end
            end
            # increment the row that we are in
            row_ind = row_ind + 1
        end
    end

    # for each day compute the mid price changes and the inter arrival times (seconds)
    for j in 1:length(days)
        # get the data for a specific day
        days_date = days[j]
        days_data = full_df[findall(x -> x == days_date, dates), :]
        # get the mid prices changes
        mid_price_changes = diff(log.(days_data[:,:midPrice]))
        # the inter arrival times (seconds)
        # get all the trading events
        days_trade_data = days_data[findall(x -> x == "TRADE", days_data[:,:eventType]),:]
        # get the ids to help with populating the full df
        trade_events_ids = days_trade_data[:,:id]
        # compute inter arrivals
        inter_arrivals = diff(datetime2unix.(days_trade_data[:,:timeStamp]))
        # add the inter-arrival data to the full_df data DataFrame
        full_df[trade_events_ids[1:(length(trade_events_ids)-1)],:interArrivals] = inter_arrivals
        # add the mid price change data to the full_df data DataFrame
        full_df[findall(x -> x == days_date, dates)[1:(size(days_data)[1]-1)], :midPriceChange] = mid_price_changes
    end

    # compute the normalized trade volume
    N = length(days)
    Tj = fill(NaN, 1, N)
    total_daily_traded_volums = fill(NaN, 1, N)

    for j in 1:length(days)
        # get the data for a specific day
        days_date = days[j]
        days_data = full_df[findall(x -> x == days_date, dates), :]
        # get all the trading events on a given day
        days_data_trade = days_data[findall(x -> x == "TRADE", days_data[:,:eventType]),:]
        # get the total number of trades in the days
        Tj[j] = size(days_data_trade)[1]
        # get the total volume traded in that day
        total_daily_traded_volums[j] = sum(days_data_trade[:,:tradeVol])
    end

    for j in 1:length(days)
        # get the data for a specific day
        days_date = days[j]
        trade_volumes = full_df[findall(x -> x == days_date, dates), :tradeVol]
        full_df[findall(x -> x == days_date, dates), :normTradeVol] = (trade_volumes)/sum(total_daily_traded_volums[j]) * (sum(Tj)/N)
    end

    return full_df

end
#---------------------------------------------------------------------------------------------------

#----- Classify trades -----#
function classifyTrades(full_df::DataFrame)

    # get all the trading dates (extract the dates from datetim object)
    dates = Date.(full_df[:,1])

    # get all the unique dates (all the days)
    days = unique(dates)

    @showprogress "Classifying trades..." for j in 1:length(days)

        # get the data for a specific day
        days_date = days[j]
        days_data = full_df[findall(x -> x == days_date, dates), :]
        #println(days_date)

        # get all the trading events on a given day
        days_data_trade = days_data[findall(x -> x == "TRADE", days_data[:,:eventType]),:]

        for i in 1:size(days_data_trade)[1]
            # get the trade event lines
            trade_event = days_data_trade[i, :]
            # apply the quote rule (compare midPrice and tradePrice)
            if trade_event[:trade] > trade_event[:midPrice]
                # get the row in the full dataframe that needs to be modified and change the trade sign
                full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["1"]
            elseif trade_event[:trade] < trade_event[:midPrice]
                # get the row in the full dataframe that needs to be modified and change the trade sign
                full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["-1"]
            elseif trade_event[:trade] == trade_event[:midPrice] # the quote rul has failed so use the tick rule
                # check if on an up or down tick and classify based on that
                if i > 1 # can only see if an uptick or downtick if there is a previous price
                    if trade_event[:trade] > days_data_trade[i-1, :trade]
                        full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["1"]
                    elseif trade_event[:trade] < days_data_trade[i-1, :trade]
                        full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["-1"]
                    elseif trade_event[:trade] == days_data_trade[i-1, :trade]
                        # now compare to the last trade that did not have the same value
                        # get all previous trades that happened including the current trade (inefficient)
                        prev_days_data_trade = days_data_trade[1:i, :]
                        # now get the last trade that was different
                        last_diff_trade_price = prev_days_data_trade[findlast(x -> x != trade_event[:trade], prev_days_data_trade[:,:trade]),:trade]
                        if trade_event[:trade] > last_diff_trade_price
                            full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["1"]
                        elseif trade_event[:trade] < last_diff_trade_price
                            full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = ["-1"]
                        else # the trade could not be classified
                            full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = [""]
                        end
                    end
                else # it was the first trade of the day and therefore cant be compared to previous trades during the day
                    full_df[findall(x -> x == trade_event[:id], full_df[:,:id]), :tradeSign] = [""]
                end
            end
        end
    end

    return full_df

end
#---------------------------------------------------------------------------------------------------

#----- Make clean TAQ data -----# 
function makeCleanTaqData(ticker::String)
    data = CSV.read("../Data/JSE/JSERAWTAQ"*ticker*".csv", DataFrame)
    println("Read in data...")
    created_full_df = createDataFrame(data) # create the dataframe that is going to hold all the info
    compact_full_df = compact(created_full_df)  # need to do compacting here
    compute_full_df = computeMidMicroArrivalsNormVol(compact_full_df) # compute columns
    classified_trades_df = classifyTrades(compute_full_df) # classify the trades
    open("../Data/JSE/L1LOB.csv", "w") do file
        println(file, "DateTime,Price,Volume,Type,Side,MidPrice,MicroPrice")
        @showprogress "Writing to L1LOB..." for i in 1:nrow(classified_trades_df)
            event = classified_trades_df[i,:]
            if event.eventType == "BID" && !(isnan(event.bid))
                !isnan(event.midPrice) ? println(file, string(event.timeStamp, ",", event.bid, ",", event.bidVol, ",Limit,1,",event.midPrice, ",", event.microPrice)) : println(file, string(event.timeStamp, ",", event.bid, ",", event.bidVol, ",Limit,1,missing,missing"))
            elseif event.eventType == "ASK" !(isnan(event.ask))
                !isnan(event.midPrice) ? println(file, string(event.timeStamp, ",", event.ask, ",", event.askVol, ",Limit,-1,",event.midPrice, ",", event.microPrice)) : println(file, string(event.timeStamp, ",", event.ask, ",", event.askVol, ",Limit,-1,missing,missing"))
            else # event is a trade
                println(file, string(event.timeStamp, ",", event.trade, ",", event.tradeVol, ",Market,", event.tradeSign, ",missing,missing"))
            end
        end
    end
end

# makeCleanTaqData("NPN")
#---------------------------------------------------------------------------------------------------