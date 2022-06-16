ENV["JULIA_COPY_STACKS"]=1
using Dates, Plots, CSV, DataFrames

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"

#----- Summary Stuff Used For Testing -----# 
function SummaryAndTestImages(messages_chnl, LOB, mid_prices, best_bids, best_asks, spreads, imbalances, chartist_ma, fundamentalist_f, ask_volumes, bid_volumes, hf_traders_vec, char_traders_vec, fun_traders_vec, messages_received, best_bid_volumes, best_ask_volumes)
    println("Messages received: " * string(length(messages_received)))

    println("Number of asks: " * string(length(LOB.asks)))
    println("Number of bids: " * string(length(LOB.bids)))

    ask_max = findmax(ask_volumes)
    println("Max Ask Volume: " *string(ask_max[1]) * " At: " * string(ask_max[2]))
    bid_max = findmax(bid_volumes)
    println("Max Bid Volume: " *string(bid_max[1]) * " At: " * string(bid_max[2]))

    println("Chartist MAs: " * join([char_traders_vec[i].λ for i in 1:parameters.Nᴸₜ], " "))

    println("HF Orders: " * string(sum(length(hf_traders_vec[i].actionTimes) for i in 1:parameters.Nᴴ)))
    println("Chartist Trades: " * join([length(char_traders_vec[i].actionTimes) for i in 1:parameters.Nᴸₜ], " "))
    println("Fun Trades: " * join([length(fun_traders_vec[i].actionTimes) for i in 1:parameters.Nᴸᵥ], " "))

    println("Fun Prices: " * join([string(fun_traders_vec[i].fₜ) for i in 1:parameters.Nᴸᵥ], " "))

    # plot the mid price 
    p1 = plot(mid_prices, label = "mid-price", title = "Prices and Traders Info", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Price")

    # add best bid and best ask
    scatter!(best_asks, label = "best asks", color = "red", markersize = 2, markerstrokewidth = 0)
    scatter!(best_bids, label = "best bids", color = "green", markersize = 2, markerstrokewidth = 0)

    # add ma of the chartist
    for i in 1:Nᴸₜ
        plot!(chartist_ma[i], label = "MA"*string(i))
    end
    for i in 1:Nᴸᵥ
        plot!(fundamentalist_f[i], label = "F"*string(i))
    end

    # New plot of the spread over time 
    p2 = plot(spreads, label = "spread", title = "Spread", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Spread")

    # plot the order imbalance
    p3 = plot(imbalances, label = "imbalance", title = "Imbalance", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Order Imbalance")

    # only prices plot
    p4 = plot(mid_prices, label = "mid-price", title = "Prices", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Price")

    # add best bid and best ask
    scatter!(best_asks, label = "best asks", color = "red", markersize = 2, markerstrokewidth = 0)
    scatter!(best_bids, label = "best bids", color = "green", markersize = 2, markerstrokewidth = 0)

    # plot the bids ask Volumes
    p5 = plot(ask_volumes, label = "asks", title = "Volumes", color="red", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Cumulative Volume")
    plot!(bid_volumes, label = "bids", color="green")

    p6 = plot(p1, p4, p3, p5, layout = 4, legend = false)

    p7 = plot(mid_prices, label = "mid-price", title = "Prices and Moving Averages", legend = :outertopright)
    xlabel!("Trade Num")
    ylabel!("Price")

    # add best bid and best ask
    scatter!(best_asks, label = "best asks", color = "red", markersize = 2, markerstrokewidth = 0)
    scatter!(best_bids, label = "best bids", color = "green", markersize = 2, markerstrokewidth = 0)

    # add ma of the chartist
    for i in 1:Nᴸₜ
        plot!(chartist_ma[i], label = "MA"*string(i) * "(" * string(round(char_traders_vec[i].λ, digits = 4)) * ")")
    end

    p8 = histogram(spreads, fillcolor = :green, linecolor = :green, xlabel = "Spread", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)

    p9 = histogram(bid_volumes, fillcolor = :green, linecolor = :green, xlabel = "Total Bid Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)

    p10 = histogram(ask_volumes, fillcolor = :green, linecolor = :green, xlabel = "Total Ask Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)

    p11 = histogram(best_bid_volumes, fillcolor = :green, linecolor = :green, xlabel = "Best bid Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)

    p12 = histogram(best_ask_volumes, fillcolor = :green, linecolor = :green, xlabel = "Best Ask Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent)

    println("best bids ", length(findall(x -> x <= mean(best_bid_volumes), best_bid_volumes)), " total length ", length(best_bid_volumes))
    println("best asks ", length(findall(x -> x <= mean(best_ask_volumes), best_ask_volumes)), " total length ", length(best_ask_volumes))

    println("mean best bid = ", mean(best_bid_volumes))
    println("mean best ask = ", mean(best_ask_volumes))

    # save plots to vis them 
    Plots.savefig(p1, path_to_files * "TestImages/prices_with_fun_char.pdf")
    Plots.savefig(p2, path_to_files * "TestImages/spread.pdf")
    Plots.savefig(p3, path_to_files * "TestImages/imbalance.pdf")
    Plots.savefig(p4, path_to_files * "TestImages/prices.pdf")
    Plots.savefig(p5, path_to_files * "TestImages/volumes.pdf")
    Plots.savefig(p6, path_to_files * "TestImages/all.pdf")
    Plots.savefig(p7, path_to_files * "TestImages/prices_MAs.pdf")
    Plots.savefig(p8, path_to_files * "TestImages/spread_dist.pdf")
    Plots.savefig(p9, path_to_files * "TestImages/total_bid_volume.pdf")
    Plots.savefig(p10, path_to_files * "TestImages/total_ask_volume.pdf")
    Plots.savefig(p11, path_to_files * "TestImages/best_bid_volume.pdf")
    Plots.savefig(p12, path_to_files * "TestImages/best_ask_volume.pdf")
end
#---------------------------------------------------------------------------------------------------

#----- Update running totals -----# 

# struct for the running totals
mutable struct RunningTotals
    best_bids::Array{Float64, 1}
    best_asks::Array{Float64, 1}
    chartist_ma::Vector{Vector{Float64}}
    fundamentalist_f::Vector{Vector{Float64}}
    imbalances::Array{Float64, 1}
    spreads::Array{Float64, 1}
    ask_volumes::Array{Float64, 1}
    bid_volumes::Array{Float64, 1}
    best_ask_volumes::Array{Float64, 1}
    best_bid_volumes::Array{Float64, 1}
end

# initialize runnning totals
function InitializeRunningTotals(Nᴸₜ::Int64, Nᴸᵥ::Int64)
    return RunningTotals(Array{Float64, 1}(),Array{Float64, 1}(), [Vector{Float64}() for i in 1:Nᴸₜ], [Vector{Float64}() for i in 1:Nᴸᵥ],Array{Float64, 1}(),Array{Float64, 1}(),Array{Float64, 1}(),Array{Float64, 1}(),Array{Float64, 1}(),Array{Float64, 1}())
end

function UpdateRunningTotals(running_totals::RunningTotals, Nᴸₜ::Int64, Nᴸᵥ::Int64, bₜ::Int64, aₜ::Int64, char_traders_vec::Vector{Chartist}, fun_traders_vec::Vector{Fundamentalist}, ρₜ::Float64, sₜ::Int64, asks::Dict{Int64, LimitOrder}, bids::Dict{Int64, LimitOrder})
    push!(running_totals.best_bids, bₜ)
    push!(running_totals.best_asks, aₜ)
    for i in 1:Nᴸₜ
        push!(running_totals.chartist_ma[i], char_traders_vec[i].p̄ₜ) 
    end
    for i in 1:Nᴸᵥ
        push!(running_totals.fundamentalist_f[i], fun_traders_vec[i].fₜ)
    end
    push!(running_totals.imbalances, ρₜ)
    push!(running_totals.spreads, sₜ)
    if length(asks) > 0
        push!(running_totals.ask_volumes, sum(order.volume for order in values(asks)))
        push!(running_totals.best_ask_volumes, sum(order.volume for order in values(asks) if order.price == aₜ))  
    end
    
    if length(bids) > 0
        push!(running_totals.bid_volumes, sum(order.volume for order in values(bids)))
        push!(running_totals.best_bid_volumes, sum(order.volume for order in values(bids) if order.price == bₜ))
    end
end

#---------------------------------------------------------------------------------------------------

#----- Write messages -----# 
function WriteMessages(initial_messages_received::Vector{String}, messages_received::Vector{String})
    # open file and write
    open(path_to_files * "/Data/CoinTossX/Raw.csv", "w") do file

        # set the Header
        println(file, "Initialization,DateTime,Type,Side,TraderMnemonic,ClientOrderId,Price,Volume")

        # add initial messages
        for message in initial_messages_received
            message_arr = split(message, "|")
            # dont write empty messages
            if message_arr[3] == ""
                continue
            end
            if length(message_arr) > 3 # trade that walked the LOB
                message_info = join(message_arr[1:2], ",")
                for trade in message_arr[3:end]
                    println(file, "INITIAL" * "," * message_info * "," * trade)
                end
            else
                println(file, "INITIAL" * "," *join(split(message, "|"), ",")) 
            end 
        end

        # add sim messages
        for message in messages_received
            message_arr = split(message, "|")
            # dont write empty messages
            if message_arr[3] == ""
                continue
            end
            if length(message_arr) > 3 # trade that walked the LOB
                message_info = join(message_arr[1:2], ",")
                for trade in message_arr[3:end]
                    println(file, "SIMULATION" * "," * message_info * "," * trade)
                end
            else
                println(file, "SIMULATION" * "," *join(split(message, "|"), ",")) 
            end
        end

    end
end
#---------------------------------------------------------------------------------------------------

#----- Write Volume Spread -----# 
function WriteVolumeSpreadData(spreads::Vector{Float64}, bid_volumes::Vector{Float64}, best_bid_volumes::Vector{Float64}, ask_volumes::Vector{Float64}, best_ask_volumes::Vector{Float64})
    spread_df = DataFrame(Spread = spreads)
    bid_vol_df = DataFrame(TotalBidVolume = bid_volumes, BestBidVolume = best_bid_volumes)
    ask_vol_df = DataFrame(TotalASKVolume = ask_volumes, BestAskVolume = best_ask_volumes)
    CSV.write(path_to_files * "/Data/RL/HistoricalDistributions/SpreadData.csv", spread_df, header = true, append = true) # keep the header as it marks the end of a single sim
    CSV.write(path_to_files * "/Data/RL/HistoricalDistributions/BidVolumeData.csv", bid_vol_df, header = true, append = true)
    CSV.write(path_to_files * "/Data/RL/HistoricalDistributions/AskVolumeData.csv", ask_vol_df, header = true, append = true)
end
#---------------------------------------------------------------------------------------------------