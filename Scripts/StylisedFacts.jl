#=
StylizedFacts:
- Julia version: 1.7.1
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie, (Some edits and additions made by Matthew Dicks)
- Function: Plots the stylised fact for JSE and CoinTossX data
- Structure:
    1. Generate stylized facts
    3. Log return sample distributions for different time resolutions
    4. Log-return and absolute log-return autocorrelation
    5. Trade sign autocorrealtion
    6. Extreme log-return percentile distribution for different time resolutions
    7. Volume-volatility correlation
    8. Depth profile
    9. Price Impact
    10. RL trade-sign autocorrelation
    11. RL absolute log-returns autocorrelation
    12. RL price impact
- Examples
    PriceImpact("Sensitivity", startTime, endTime); PriceImpact("JSE", startTime, endTime)
    StylizedFacts("CoinTossX", startTime, endTime); StylizedFacts("JSE", startTime, endTime)
=#
using Distributions, CSV, Plots, StatsPlots, Dates, StatsBase, DataFrames, Plots.PlotMeasures
import Statistics.var
#---------------------------------------------------------------------------------------------------

# set working directory (the path to the Scripts/StylisedFacts.jl file)
path_to_folder = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/Scripts"
cd(path_to_folder)
include("Moments.jl")

#----- Generate stylized facts -----#
function StylizedFacts(exchange::String, l1lobPath::String, startTime::DateTime, endTime::DateTime; format::String = "pdf")
    println("Computing stylized facts")
    println("Reading in data...")
    suffix = ""
    l1lobPath == "L1LOB" ? suffix = "" : suffix = string(split(l1lobPath, "/")[end][6:end])
    occursin("/", l1lobPath) ? filepath = join(split(l1lobPath, "/")[1:end-1], "/") : filepath = "" 
    if !(isdir("../Images/" * exchange * "/" * filepath))
        mkdir("../Images/" * exchange * "/" * filepath)
    end
    if exchange == "CoinTossX"
        data = CSV.File(string("../Data/" * exchange * "/" * l1lobPath * ".csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
    else
        data = CSV.File(string("../Data/" * exchange * "/" * l1lobPath * ".csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
        filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, data)
    end
    data.Date = Date.(data.DateTime)
    uniqueDays = unique(data.Date)
    logreturns = map(day -> diff(log.(skipmissing(data[searchsorted(data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x)
    LogReturnDistribution(exchange, suffix, filepath, logreturns; format = format)
    println("Log-return distribution complete")
    LogReturnAutocorrelation(exchange, suffix, filepath, logreturns, 500; format = format)
    println("Log-return auto-correaltion complete")
    AbsLogReturnAutocorrelation(exchange, suffix, filepath, logreturns, 500; format = format)
    println("Absolute Log-return auto-correaltion complete")
    TradeSignAutocorrelation(exchange, suffix, filepath, data, 500; format = format)
    println("Trade sign auto-correaltion complete")
    ExtremeLogReturnPercentileDistribution(exchange, suffix, filepath, logreturns; format = format)
    println("Extreme log-return percentile distributions compelete")
    println("Stylised facts complete")
    trades = filter(x -> x.Type == :Market, data)
    println("Trade Volume: " * string(sum(trades[:,:Volume])))
end
#---------------------------------------------------------------------------------------------------

#----- Log return sample distributions for different time resolutions -----#
function LogReturnDistribution(exchange::String, suffix::String, filepath::String, logreturns::Vector{Float64}; format::String = "pdf")
    color = exchange == "CoinTossX" ? :green : (exchange == "JSE" ? :purple : :orange)
    NormalDistribution = Distributions.fit(Normal, logreturns)
    distribution = histogram(logreturns, normalize = :pdf, fillcolor = color, linecolor = color, title = "", xlabel = "Log returns", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent, ylim = (0, 3000), xlim = (-0.005, 0.005), fontfamily="Computer Modern")
    plot!(distribution, NormalDistribution, line = (:black, 2), label = "Fitted Normal")
    qqplot!(distribution, Normal, logreturns, xlabel = "Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (color, stroke(color), 3), legend = false, inset = (1,bbox(0.125, 0.065, 0.37, 0.435)), subplot = 2, title = "Normal QQ-plot", titlefontsize = 7) # standard bbox(0.125, 0.03, 0.37, 0.4), With title bbox(0.125, 0.065, 0.37, 0.435)
    savefig(distribution, string("../Images/" * exchange * "/" * filepath * "/Log-ReturnDistribution" * suffix * ".", format))
end
#---------------------------------------------------------------------------------------------------

#----- Log-return and absolute log-return autocorrelation -----#
function LogReturnAutocorrelation(exchange::String, suffix::String, filepath::String, logreturns::Vector{Float64}, lag::Int64; format::String = "pdf")
    color = exchange == "CoinTossX" ? :green : (exchange == "JSE" ? :purple : :orange)
    lag = length(logreturns) - 1
    autoCorr = autocor(logreturns, 1:lag; demean = false)
    absAutoCorr = autocor(abs.(logreturns), 1:lag; demean = false)
    autoCorrPlot = plot(autoCorr, seriestype = [:sticks, :scatter], marker = (color, stroke(color), 3), linecolor = :black, title = "", xlabel = "Lag", ylabel = "Autocorrelation", legend = false, ylim = (-0.4, 0.2), fontfamily="Computer Modern")
    plot!(autoCorrPlot, [1.96 / sqrt(length(logreturns)), -1.96 / sqrt(length(logreturns))], seriestype = :hline, line = (:dash, :black, 2))
    plot!(autoCorrPlot, absAutoCorr, seriestype = :scatter, marker = (color, stroke(color), 3), legend = false, xlabel = "Lag", ylabel = "Autocorrelation", inset = (1, bbox(0.62, 0.47, 0.4, 0.37, :top)), subplot = 2, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, title = "Absolute log-return autocorrelation", titlefontsize = 7, ylim = (-0.1, 0.7))
    savefig(autoCorrPlot, string("../Images/" * exchange * "/" * filepath * "/Log-ReturnAutocorrelation" * suffix * ".", format))
end
#---------------------------------------------------------------------------------------------------

#----- Absolute log-return autocorrelation -----#
function AbsLogReturnAutocorrelation(exchange::String, suffix::String, filepath::String, logreturns::Vector{Float64}, lag::Int64; format::String = "pdf")
    color = exchange == "CoinTossX" ? :green : (exchange == "JSE" ? :purple : :orange)
    lag = length(logreturns) - 1
    absAutoCorr = autocor(abs.(logreturns), 1:lag; demean = false)
    autoCorrPlot = plot(absAutoCorr, seriestype = :scatter, marker = (color, stroke(color), 3), legend = false, xlabel = "Lag", ylabel = "Autocorrelation", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", ylim = (-0.1, 0.7), fontfamily="Computer Modern")
    plot!(autoCorrPlot, [1.96 / sqrt(length(logreturns)), -1.96 / sqrt(length(logreturns))], seriestype = :hline, line = (:dash, :black, 2))
    savefig(autoCorrPlot, string("../Images/" * exchange * "/" * filepath * "/AbsLog-ReturnAutocorrelation" * suffix * "_V" * string(V) * "_S" * string(numT) * "_430.", format))
end
#---------------------------------------------------------------------------------------------------

#----- Trade sign autocorrealtion -----#
function TradeSignAutocorrelation(exchange::String, suffix::String, filepath::String, data::DataFrame, lag::Int64; format::String = "pdf")
    color = exchange == "CoinTossX" ? :green : (exchange == "JSE" ? :purple : :orange)
    tradeSigns = data[findall(x -> x == :Market, data.Type), :Side]
    lag = length(tradeSigns) - 1
    autoCorr = autocor(tradeSigns, 1:lag; demean = false)

    # plot the positive autocorrelations on a log-log scale and estimate the tail index\
    if exchange == "JSE"
        xₘᵢₙ = minimum(autoCorr[findall(x -> x > 0, autoCorr)])
        α = 1 + length(autoCorr[findall(x -> x > 0, autoCorr)]) / sum(log.(autoCorr[findall(x -> x > 0, autoCorr)] ./ xₘᵢₙ))
        println()
        println(string(exchange, " tail-index estimate"))
        println(α)
        println()
        p = plot(autoCorr[findall(x -> x > 0, autoCorr)], xscale = :log10, yscale = :log10, legend = false, xlabel = "Lag", ylabel = "Autocorrelation", linecolor = color, title = "Log-Log-scale order-flow autocorrelation", fontfamily="Computer Modern")
        savefig(p, string("../Images/" * exchange * "/Trade-SignAutocorrelationlog-log" * suffix * ".", format))
    end

    autoCorrPlot = plot(autoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = false, title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), fontfamily="Computer Modern")
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(tradeSigns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(tradeSigns))], seriestype = :hline, line = (:dash, :black, 2))
    plot!(autoCorrPlot, autoCorr, xscale = :log10, inset = (1, bbox(0.58, 0.1, 0.4, 0.4)), subplot = 2, legend = false, xlabel = "Lag", guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, ylabel = "Autocorrelation", linecolor = color, title = "Log-scale order-flow autocorrelation", titlefontsize = 7) #  ", L"(\log_{10})
    savefig(autoCorrPlot, string("../Images/" * exchange * "/" * filepath * "/Trade-SignAutocorrelation" * suffix * ".", format))
end
#---------------------------------------------------------------------------------------------------

#----- Extreme log-return percentile distribution for different time resolutions -----#
function ExtremeLogReturnPercentileDistribution(exchange::String, suffix::String, filepath::String, logreturns::Vector{Float64}; format::String = "pdf")
    color = exchange == "CoinTossX" ? :green : (exchange == "JSE" ? :purple : :orange)
    upperobservations = logreturns[findall(x -> x >= quantile(logreturns, 0.95), logreturns)]; lowerobservations = -logreturns[findall(x -> x <= quantile(logreturns, 0.05), logreturns)]
    sort!(upperobservations); sort!(lowerobservations)
    upperxₘᵢₙ = minimum(upperobservations); lowerxₘᵢₙ = minimum(lowerobservations)
    upperα = 1 + length(upperobservations) / sum(log.(upperobservations ./ upperxₘᵢₙ)); lowerα = 1 + length(lowerobservations) / sum(log.(lowerobservations ./ lowerxₘᵢₙ))
    upperTheoreticalQuantiles = map(i -> (1 - (i / length(upperobservations))) ^ (-1 / (upperα - 1)) * upperxₘᵢₙ, 1:length(upperobservations)); lowerTheoreticalQuantiles = map(i -> (1 - (i / length(lowerobservations))) ^ (-1 / (lowerα - 1)) * lowerxₘᵢₙ, 1:length(lowerobservations))
    extremePercentileDistributionPlot = density(upperobservations, seriestype = [:scatter, :line], marker = (color, stroke(color), :utriangle), linecolor = color, title = "", xlabel = string("Log return extreme percentiles"), ylabel = "Density", label = string("Upper percentiles - α = ", round(upperα, digits = 3)), legend = :topright, fg_legend = :transparent, fontfamily="Computer Modern")
    density!(extremePercentileDistributionPlot, lowerobservations, seriestype = [:scatter, :line], marker = (color, stroke(color), :dtriangle), linecolor = color, label = string("Lower percentiles - α = ", round(lowerα, digits = 3)))
    plot!(extremePercentileDistributionPlot, hcat(upperTheoreticalQuantiles, upperTheoreticalQuantiles), hcat(upperobservations, upperTheoreticalQuantiles), scale = :log10, seriestype = [:scatter :line], inset = (1, bbox(0.42, 0.22, 0.56, 0.53, :top)), subplot = 2, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, legend = :none, xlabel = "Power-Law Theoretical Quantiles", ylabel = "Sample Quantiles", linecolor = :black, marker = (color, stroke(color), 3, [:utriangle :none]), fg_legend = :transparent, title = "Power-Law QQ-plot", titlefontsize = 7)
    plot!(extremePercentileDistributionPlot, [lowerTheoreticalQuantiles lowerTheoreticalQuantiles], [lowerobservations lowerTheoreticalQuantiles], seriestype = [:scatter :line], subplot = 2, linecolor = :black, marker = (color, stroke(color), 3, [:dtriangle :none]))
    savefig(extremePercentileDistributionPlot, string("../Images/" * exchange * "/" * filepath * "/ExtremeLog-ReturnPercentilesDistribution" * suffix * ".", format))
end
#---------------------------------------------------------------------------------------------------

#----- Depth profile -----#
function DepthProfile(exchange::String, depthProfilePath::String; format::String = "pdf")
    println("Computing depth profiles")
    println("Reading in data...")
    profile = CSV.File(string("../Data/" * exchange * "/" * depthProfilePath * ".csv"), header = false) |> DataFrame |> Matrix{Union{Missing, Int64}}
    μ = map(i -> mean(skipmissing(profile[:, i])), 1:size(profile, 2))
    depthProfile = plot(-(1:7), μ[1:7], seriestype = [:scatter, :line], marker = (:blue, stroke(:blue), :utriangle), linecolor = :blue, label = ["" "Bid profile"], title = "", xlabel = "Price level of limit orders (<0: bids; >0: asks)", ylabel = "Volume", legend = :topleft, fg_legend = :transparent, right_margin = 15mm, fontfamily="Computer Modern")#, yscale = :log10
    plot!(twinx(), 1:7, μ[8:14], seriestype = [:scatter, :line], marker = (:red, stroke(:red), :dtriangle), linecolor = :red, label = ["" "Ask profile"], legend = :topright, fg_legend = :transparent)
    suffix = ""
    occursin("/", depthProfilePath) ? filepath = join(split(depthProfilePath, "/")[1:end-1], "/") : filepath = "" 
    savefig(depthProfile, string("../Images/" * exchange * "/" * filepath * "/DepthProfile" * suffix * ".", format))
    println("Depth profile visualization complete")
end
#---------------------------------------------------------------------------------------------------

#----- Extract price-impact data -----#
function PriceImpact(exchange::String, l1lobPath::String, startTime::DateTime, endTime::DateTime; format::String = "pdf")
    println("Computing price impact")
    println("Reading in data...")
    if exchange == "CoinTossX"
        data = CSV.File(string("../Data/" * exchange * "/" * l1lobPath * ".csv"), drop = [:MicroPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
    else
        data = CSV.File(string("../Data/" * exchange * "/" * l1lobPath * ".csv"), drop = [:MicroPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
        filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, data)
    end
    buyerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}()); sellerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}())
    days = unique(x -> Date(x), data.DateTime)
    tradeIndeces = findall(x -> x == :Market, data.Type)
    totalTradeCount = length(tradeIndeces)
    for day in days
        dayTradeIndeces = tradeIndeces[searchsorted(data[tradeIndeces, :DateTime], day, by = Date)]
        dayVolume = sum(data[dayTradeIndeces, :Volume])
        for index in dayTradeIndeces
            if index == 1 # can't compute mid price change if it is the first thing that happened
                continue
            end
            midPriceBeforeTrade = data[index - 1, :MidPrice]; midPriceAfterTrade = data[index + 1, :MidPrice]
            Δp = log(midPriceAfterTrade) - log(midPriceBeforeTrade)
            ω = (data[index, :Volume] / dayVolume) * (totalTradeCount / length(days))
            if !ismissing(Δp) && !ismissing(ω)
                data[index, :Side] == 1 ? push!(buyerInitiated, (Δp, ω)) : push!(sellerInitiated, (-Δp, ω))
            end
        end
    end
    filter!(x -> !isnan(x.Impact) && !isnan(x.NormalizedVolume) && x.Impact > 0, buyerInitiated); filter!(x -> !isnan(x.Impact) && !isnan(x.NormalizedVolume) && x.Impact > 0, sellerInitiated)
    normalisedVolumeBins = 10 .^ (range(-1, 1, length = 21))
    Δp = fill(NaN, (length(normalisedVolumeBins), 2)); ω = fill(NaN, (length(normalisedVolumeBins), 2)) # Column 1 is buy; column 2 is sell
    for i in 2:length(normalisedVolumeBins)
        binIndeces = (findall(x -> normalisedVolumeBins[i - 1] < x <= normalisedVolumeBins[i], buyerInitiated.NormalizedVolume), findall(x -> normalisedVolumeBins[i - 1] < x <= normalisedVolumeBins[i], sellerInitiated.NormalizedVolume))
        if !isempty(first(binIndeces))
            Δp[i - 1, 1] = mean(buyerInitiated[first(binIndeces), :Impact]); ω[i, 1] = mean(buyerInitiated[first(binIndeces), :NormalizedVolume])
        end
        if !isempty(last(binIndeces))
            Δp[i - 1, 2] = mean(sellerInitiated[last(binIndeces), :Impact]); ω[i, 2] = mean(sellerInitiated[last(binIndeces), :NormalizedVolume])
        end
    end
    exchange == "CoinTossX" ? title = "ABM" : title = "JSE"
    indeces = findall(vec(any(x -> !isnan(x), Δp, dims = 2) .* any(x -> !isnan(x), ω, dims = 2)))
    priceImpact = plot(ω[2:(end-3), :], Δp[2:(end-3), :], scale = :log10, seriestype = [:scatter, :line], markershape = [:utriangle :dtriangle], markercolor = [:blue :red], markerstrokecolor = [:blue :red], markersize = 3, linecolor = [:blue :red], xlabel = "ω*", ylabel = "Δp*", label = ["" "" "Buyer initiated" "Seller initiated"], legend = :topleft, fg_legend = :transparent, title = title, fontfamily="Computer Modern")
    suffix = ""
    l1lobPath == "L1LOB" ? suffix = "" : suffix = string(split(l1lobPath, "/")[end][6:end])
    occursin("/", l1lobPath) ? filepath = join(split(l1lobPath, "/")[1:end-1], "/") : filepath = "" 
    savefig(priceImpact, string("../Images/" * exchange * "/" * filepath * "/PriceImpact" * suffix * ".", format))
    println("Price impact complete")
end
#---------------------------------------------------------------------------------------------------

#----- Modified Cox confidence intervals for log normal distribution -----#
function LogNormalCI(yBar::Float64, s::Float64, n::Int64)
    theta_hat = exp(yBar + (s^2)/2)
    lower = exp(yBar + (s^2)/2 - 2.02 * sqrt((s^2)/n + (s^4)/(2 * (n - 1))))
    upper = exp(yBar + (s^2)/2 + 2.02 * sqrt((s^2)/n + (s^4)/(2 * (n - 1))))
    return round(theta_hat, digits = 3), round(lower, digits = 3), round(upper, digits = 3)
end
#---------------------------------------------------------------------------------------------------

#----- Get the ADV (traded and limit) volume for CTX -----#
function ADVCoinTossX(;format = "pdf")
    traded_volumes = Vector{Int64}()
    limit_volumes = Vector{Int64}()
    for i in 1:100
        filename = "Raw" * string(i)
        orders = CSV.File(string("../Data/CoinTossX/ADV/SimulationData/" * filename * ".csv"), types = Dict(:Initialization => Symbol, :ClientOrderId => Int64, :DateTime => DateTime, :Price => Int64, :Volume => Int64, :Side => Symbol, :Type => Symbol, :TraderMnemonic => Symbol), dateformat = "yyyy-mm-ddTHH:MM:SS.s") |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        replace!(orders.Type, :New => :Limit); # Rename Types # orders.Type[findall(x -> x == 0, orders.Price)] .= :Market 
        orders.ClientOrderId[findall(x -> x == :Cancelled, orders.Type)] .*= -1 # convert -id given by CTX to id
        DataFrames.rename!(orders, [:ClientOrderId => :OrderId, :TraderMnemonic => :Trader])
        push!(limit_volumes, sum(orders[findall(x -> x == :Limit, orders.Type),:Volume]))
        push!(traded_volumes, sum(orders[findall(x -> x == :Trade, orders.Type),:Volume]))
    end
    color = "green"
    
    # trade volume distribution
    LogNormalDistributionTradeVol = Distributions.fit(LogNormal, traded_volumes)
    fitted_trade_distribution = histogram(traded_volumes, normalize = :pdf, fillcolor = color, linecolor = color, title = "", xlabel = "Trade Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent, fontfamily="Computer Modern")
    plot!(fitted_trade_distribution, LogNormalDistributionTradeVol, line = (:black, 2), label = "Fitted Normal")
    qqplot!(fitted_trade_distribution, LogNormal, traded_volumes, xlabel = "Log-Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (color, stroke(color), 3), legend = false, inset = (1,bbox(0.125, 0.065, 0.37, 0.435)), subplot = 2, title = "Log-Normal QQ-plot", titlefontsize = 7) # standard bbox(0.125, 0.03, 0.37, 0.4), With title bbox(0.125, 0.065, 0.37, 0.435)
    savefig(fitted_trade_distribution, string("../Images/CoinTossX/TradeVolumeDistribution.", format))
    theta_hat_trade, lower_trade, upper_trade = LogNormalCI(LogNormalDistributionTradeVol.μ, LogNormalDistributionTradeVol.σ, length(traded_volumes))

    # limit volume distribution
    LogNormalDistributionLimitVol = Distributions.fit(LogNormal, limit_volumes)
    fitted_limit_distribution = histogram(limit_volumes, normalize = :pdf, fillcolor = color, nbins = 20, linecolor = color, title = "", xlabel = "Limit Volume", ylabel = "Probability Density", label = "Empirical", legendtitle = "Distribution", legend = :topright, legendfontsize = 5, legendtitlefontsize = 7, fg_legend = :transparent, fontfamily="Computer Modern")
    plot!(fitted_limit_distribution, LogNormalDistributionLimitVol, line = (:black, 2), label = "Fitted Normal")
    qqplot!(fitted_limit_distribution, LogNormal, limit_volumes, xlabel = "Log-Normal theoretical quantiles", ylabel = "Sample quantiles", linecolor = :black, guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, marker = (color, stroke(color), 3), legend = false, inset = (1,bbox(0.625, 0.265, 0.37, 0.435)), subplot = 2, title = "Log-Normal QQ-plot", titlefontsize = 7) # standard bbox(0.125, 0.03, 0.37, 0.4), With title bbox(0.125, 0.065, 0.37, 0.435)
    savefig(fitted_limit_distribution, string("../Images/CoinTossX/LimitVolumeDistribution.", format))
    theta_hat_limit, lower_limit, upper_limit = LogNormalCI(LogNormalDistributionLimitVol.μ, LogNormalDistributionLimitVol.σ, length(limit_volumes))

    # write results to a file
    adv_res = DataFrame(OrderedDict(:Type => ["Trade", "Limit"], :Lower => [lower_trade, lower_limit], :ThetaHat => [theta_hat_trade, theta_hat_limit], :Upper => [upper_trade, upper_limit], :ADV => [round(mean(traded_volumes), digits = 3), round(mean(limit_volumes), digits = 3)]))
    CSV.write("../Data/CoinTossX/ADV/ADVResults.csv", adv_res)
end
# ADVCoinTossX()
#---------------------------------------------------------------------------------------------------

#----- Get the traded and limit volume for JSE -----#
function ADVJSE()
    date = DateTime("2019-07-08")
    startTime = date + Hour(9) + Minute(1)
    endTime = date + Hour(16) + Minute(50)
    filename = "JSERAWTAQNPN"
    orders = CSV.read("../Data/JSE/JSERAWTAQNPN.csv", types = Dict(:type => Symbol), DataFrame)
    filter!(x -> startTime <= x.times && x.times < endTime, orders)
    adv_res = DataFrame(OrderedDict(:Type => ["Trade", "Limit"], :Volume => [sum(orders[findall(x -> x == :TRADE, orders.type),:size]), sum(orders[findall(x -> x == :ASK || x == :BID, orders.type),:size])]))
    CSV.write("../Data/JSE/ADVResults.csv", adv_res)
end
# ADVJSE()
#---------------------------------------------------------------------------------------------------

#----- RL agents trade-sign autocorrelations (single figure, final agents) -----#
function RLTradeSignAutocorrelation(stateSpaceSizes::Vector{Int64}, initialInventories::Vector{Int64}; format::String = "pdf")
    count = 1
    colors = [:green,:red,:blue,:orange,:brown,:magenta]
    autoCorrPlot = nothing
    tradeSigns = nothing
    for s in stateSpaceSizes
        for v in initialInventories
            color = colors[count]
            l1lobPath = "/alpha0.1_iterations1000_V" * string(v) * "_S" * string(s) * "_430/L1LOBRLIteration1000"
            data = CSV.File(string("../Data/CoinTossX/" * l1lobPath * ".csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
            data.Date = Date.(data.DateTime)
            uniqueDays = unique(data.Date)
            tradeSigns = data[findall(x -> x == :Market, data.Type), :Side]
            lag = length(tradeSigns) - 1
            autoCorr = autocor(tradeSigns, 1:lag; demean = false)
            if count == 1
                autoCorrPlot = plot(autoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$", title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), fontfamily="Computer Modern")
                plot!(autoCorrPlot, autoCorr, xscale = :log10, inset = (1, bbox(0.58, 0.1, 0.4, 0.4)), legend = false, subplot = 2, xlabel = "Lag", guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, ylabel = "Autocorrelation", linecolor = color, title = "Log-scale order-flow autocorrelation", titlefontsize = 7, ylim = (-0.1, 0.5), fontfamily="Computer Modern")
            else
                plot!(autoCorrPlot, autoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$", title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), fontfamily="Computer Modern")
                plot!(autoCorrPlot[2], autoCorr, xscale = :log10, legend = false, xlabel = "Lag", guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, ylabel = "Autocorrelation", linecolor = color, title = "Log-scale order-flow autocorrelation", titlefontsize = 7, ylim = (-0.1, 0.4))
            end
            count += 1
        end
    end
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(tradeSigns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(tradeSigns))], seriestype = :hline, line = (:dash, :black, 2), label = "")
    savefig(autoCorrPlot, string("../Images/CoinTossX/RLTradeSignAutocorrelation_430.", format))
    println("RL agents trade-sign autocorrelation complete")
end
# stateSpaceSizes = [5,10]
# initialInventories = [50, 100, 200]
# RLTradeSignAutocorrelation(stateSpaceSizes, initialInventories)
#---------------------------------------------------------------------------------------------------

#----- RL agents absolute log-return autocorrelations (single figure, final agents) -----#
function RLAbsLogReturnAutocorrelation(stateSpaceSizes::Vector{Int64}, initialInventories::Vector{Int64}; format::String = "pdf")
    count = 1
    colors = [:green,:red,:blue,:orange,:brown,:magenta]
    autoCorrPlot = nothing
    logreturns = nothing
    for s in stateSpaceSizes
        for v in initialInventories
            color = colors[count]
            l1lobPath = "/alpha0.1_iterations1000_V" * string(v) * "_S" * string(s) * "_430/L1LOBRLIteration1000"
            data = CSV.File(string("../Data/CoinTossX/" * l1lobPath * ".csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
            data.Date = Date.(data.DateTime)
            uniqueDays = unique(data.Date)
            logreturns = map(day -> diff(log.(skipmissing(data[searchsorted(data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x)
            lag = length(logreturns) - 1
            absAutoCorr = autocor(abs.(logreturns), 1:lag; demean = false)
            if count == 1
                autoCorrPlot = plot(absAutoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$", title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), fontfamily="Computer Modern")
            else
                plot!(autoCorrPlot, absAutoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$", title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), fontfamily="Computer Modern")
            end
            count += 1
        end
    end
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(logreturns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(logreturns))], seriestype = :hline, line = (:dash, :black, 2), label = "")
    savefig(autoCorrPlot, string("../Images/CoinTossX/RLAbsLog-ReturnAutocorrelation_430.", format))
    println("RL agents trade-sign autocorrelation complete")
end
# stateSpaceSizes = [5,10]
# initialInventories = [50, 100, 200]
# RLAbsLogReturnAutocorrelation(stateSpaceSizes, initialInventories)
#---------------------------------------------------------------------------------------------------

#----- Extract price-impact data -----#
function RLPriceImpact(stateSpaceSizes::Vector{Int64}, initialInventories::Vector{Int64}; format::String = "pdf")
    println("Computing RL agents price impact")
    println("Reading in data...")
    count = 1
    colors = [:green,:red,:blue,:orange,:brown,:magenta]
    priceImpactBuy = nothing
    priceImpactSell = nothing
    for s in stateSpaceSizes
        for v in initialInventories
            color = colors[count]
            l1lobPath = "/alpha0.1_iterations1000_V" * string(v) * "_S" * string(s) * "_430/L1LOBRLIteration1000"
            data = CSV.File(string("../Data/CoinTossX/" * l1lobPath * ".csv"), drop = [:MicroPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
            buyerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}()); sellerInitiated = DataFrame(Impact = Vector{Float64}(), NormalizedVolume = Vector{Float64}())
            days = unique(x -> Date(x), data.DateTime)
            tradeIndeces = findall(x -> x == :Market, data.Type)
            totalTradeCount = length(tradeIndeces)
            for day in days
                dayTradeIndeces = tradeIndeces[searchsorted(data[tradeIndeces, :DateTime], day, by = Date)]
                dayVolume = sum(data[dayTradeIndeces, :Volume])
                for index in dayTradeIndeces
                    if index == 1 # can't compute mid price change if it is the first thing that happened
                        continue
                    end
                    midPriceBeforeTrade = data[index - 1, :MidPrice]; midPriceAfterTrade = data[index + 1, :MidPrice]
                    Δp = log(midPriceAfterTrade) - log(midPriceBeforeTrade)
                    ω = (data[index, :Volume] / dayVolume) * (totalTradeCount / length(days))
                    if !ismissing(Δp) && !ismissing(ω)
                        data[index, :Side] == 1 ? push!(buyerInitiated, (Δp, ω)) : push!(sellerInitiated, (-Δp, ω))
                    end
                end
            end
            filter!(x -> !isnan(x.Impact) && !isnan(x.NormalizedVolume) && x.Impact > 0, buyerInitiated); filter!(x -> !isnan(x.Impact) && !isnan(x.NormalizedVolume) && x.Impact > 0, sellerInitiated)
            normalisedVolumeBins = 10 .^ (range(-1, 1, length = 21))
            Δp = fill(NaN, (length(normalisedVolumeBins), 2)); ω = fill(NaN, (length(normalisedVolumeBins), 2)) # Column 1 is buy; column 2 is sell
            for i in 2:length(normalisedVolumeBins)
                binIndeces = (findall(x -> normalisedVolumeBins[i - 1] < x <= normalisedVolumeBins[i], buyerInitiated.NormalizedVolume), findall(x -> normalisedVolumeBins[i - 1] < x <= normalisedVolumeBins[i], sellerInitiated.NormalizedVolume))
                if !isempty(first(binIndeces))
                    Δp[i - 1, 1] = mean(buyerInitiated[first(binIndeces), :Impact]); ω[i, 1] = mean(buyerInitiated[first(binIndeces), :NormalizedVolume])
                end
                if !isempty(last(binIndeces))
                    Δp[i - 1, 2] = mean(sellerInitiated[last(binIndeces), :Impact]); ω[i, 2] = mean(sellerInitiated[last(binIndeces), :NormalizedVolume])
                end
            end
            indeces = findall(vec(any(x -> !isnan(x), Δp, dims = 2) .* any(x -> !isnan(x), ω, dims = 2)))
            if count == 1
                priceImpactBuy = plot(ω[2:(end-3), 1], Δp[2:(end-3), 1], scale = :log10, seriestype = [:scatter, :line], markershape = [:utriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = ["" raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$"], legend = :topleft, fg_legend = :transparent, title = "Buyer initiated", fontfamily="Computer Modern")
                priceImpactSell = plot(ω[2:(end-3), 2], Δp[2:(end-3), 2], scale = :log10, seriestype = [:scatter, :line], markershape = [:dtriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = ["" raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$"], legend = :topleft, fg_legend = :transparent, title = "Seller initiated", fontfamily="Computer Modern")
            else
                plot!(priceImpactBuy, ω[2:(end-3), 1], Δp[2:(end-3), 1], scale = :log10, seriestype = [:scatter, :line], markershape = [:utriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = ["" raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$"], legend = :topleft, fg_legend = :transparent, title = "Buyer initiated", fontfamily="Computer Modern")
                plot!(priceImpactSell, ω[2:(end-3), 2], Δp[2:(end-3), 2], scale = :log10, seriestype = [:scatter, :line], markershape = [:dtriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = ["" raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(s) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * v) * raw"}$"], legend = :topleft, fg_legend = :transparent, title = "Seller initiated", fontfamily="Computer Modern")
            end
            count += 1
        end
    end
    savefig(priceImpactBuy, string("../Images/CoinTossX/RLPriceImpactBuyerInitiated_430.", format))
    savefig(priceImpactSell, string("../Images/CoinTossX/RLPriceImpactSellerInitiated_430.", format))
    println("RL agents price impact complete")
end
# stateSpaceSizes = [5,10]
# initialInventories = [50, 100, 200]
# RLPriceImpact(stateSpaceSizes, initialInventories)
#---------------------------------------------------------------------------------------------------

# # make sure these are the same as the ones used in the sensitivity analysis
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50) 

# StylizedFacts("JSE", "L1LOB", startTime, endTime)
# PriceImpact("JSE", "L1LOB", startTime, endTime)
# StylizedFacts("CoinTossX", "L1LOB", startTime, endTime) # "/alpha0.1_iterations1000_V200_S10_430/L1LOBRLIteration1000" numT = , V = 
# PriceImpact("CoinTossX", "L1LOB", startTime, endTime)
# DepthProfile("CoinTossX", "DepthProfileData") # "/alpha0.1_iterations1000_V50_S5_430/DepthProfileDataRLIteration1000"
