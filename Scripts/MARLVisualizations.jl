#=
MARLVisualisations:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie
- Function: Visualise the data generated from training the MARL agents
- Structure:
    1. 
- Examples:
    1. 
=#
ENV["JULIA_COPY_STACKS"]=1
using Dates, DataFrames, DynamicalSystems, CSV, Plots, Statistics, DataStructures, JLD, Plots.PlotMeasures, LaTeXStrings, Distributions, Random, StatsBase, KernelDensity, Interpolations, StatsPlots, ColorSchemes, Measures

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/Moments.jl"); 

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/Moments.jl"); include(path_to_files * "DataCleaning/CoinTossX.jl")

#----- Create combination's file path -----# 
function CreateCombinations(types::Vector{Int64}, numType1Buyers::Int64, numType1Sellers::Int64, numType2Buyers::Int64, numType2Sellers::Int64)
    raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
    if 1 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
    end
    if 2 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
    end
    raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
    if 2 in types
        raw_data_dir = raw_data_dir * "_delta_3_6"
    end
    return raw_data_dir
end
#---------------------------------------------------------------------------------------------------

#----- Clean all raw message data into L1LOB format -----# 
function CleanAllAgentData(cleaningDirectory::String, types::Vector{Int64}, numType1Buyers::Int64, numType1Sellers::Int64, numType2Buyers::Int64, numType2Sellers::Int64, iterations::Int64, iterations_per_write::Int64)

    # create folder path to raw data
    raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
    if 1 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
    end
    if 2 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
    end
    raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
    if 2 in types
        raw_data_dir = raw_data_dir * "_delta_3_6"
    end

    # for each iteration
    for i in ([1;range(1,Int(iterations/iterations_per_write)) * iterations_per_write])

        # create file path
        file_path = raw_data_dir * "/RawRLIteration" * string(i) * ".csv"

        # copy file to cleaning directory
        run(`cp $file_path $cleaningDirectory`)

        # clean into L1LOB, TAQ format, and DepthProfileFormat
        CleanData("RawRLIteration" * string(i), initialization = false)

        # move L1LOB back to folder 
        l1lob_file_path = cleaningDirectory * "L1LOBRLIteration" * string(i) * ".csv"
        run(`mv $l1lob_file_path $raw_data_dir`)

        # delete TAQ, DepthProfile and moved raw data
        depth_file_path = cleaningDirectory * "/DepthProfileDataRLIteration" * string(i) *".csv"
        taq_file_path =  cleaningDirectory * "/TAQRLIteration" * string(i) *".csv"
        raw_file_path = cleaningDirectory * "/RawRLIteration" * string(i) *".csv"
        run(`rm $depth_file_path`)
        run(`rm $taq_file_path`)
        run(`rm $raw_file_path`)

    end

end
# types = [2]
# numType1Buyers = 0
# numType1Sellers = 0
# numType2Buyers = 5
# numType2Sellers = 5
# cleaningDirectory = path_to_files * "/Data/CoinTossX/"
# iterations = 1000
# iterations_per_write = 10
# CleanAllAgentData(cleaningDirectory, types, numType1Buyers, numType1Sellers, numType2Buyers, numType2Sellers, iterations, iterations_per_write)
#---------------------------------------------------------------------------------------------------

#----- Reward convergence for all buyer seller RL agents -----# 
function Winsorise(data::Vector{Float64}, tol::Float64)
    return data[findall(x -> x < quantile(data, 1-tol) && x > quantile(data, tol), data)]
end

function ExponentialMovingAverage(data::Vector{Float64}, alpha::Float64)
    s = Vector{Float64}()
    push!(s, data[1])
    for i in 2:length(data)
        push!(s, alpha * data[i] + (1-alpha) * s[i-1])
    end
    return s
end

function RewardConvergence(marl_data, tol, alpha)
    plots = Dict()
    for (j, key) in enumerate(keys(marl_data))
        res = marl_data[key]
        num_episodes = length(res)
        labels = Set{String}()
        for (k, agent) in enumerate(keys(res[1]))
            if "AgentType" in keys(res[k][agent])
                agent_la_type = res[k][agent]["AgentType"]
            else
                agent_la_type = "Type1"
            end
            agent_la_type == "Type1" ? agent_la_type = "I" : agent_la_type = "II"
            res[k][agent]["ActionType"] == "Buy" ? agent_action_type = "+" : agent_action_type = "-"
            fields = split(key, "_")
            label = fields[1]
            if length(fields) > 1
                fields[1][1] != '5' ? label = "{" * label * "," * fields[2] * "}" : label = label * raw" $\bigcup$ " * fields[2]
                label = agent_la_type * agent_action_type * raw"$\in$" * label
            else
                fields[1][1] != '5' ? label = agent_la_type * agent_action_type * raw"$\in$" * "{" * label * "}" : label = agent_la_type * agent_action_type * raw"$\in$" * label
            end
            if label in labels
                continue
            else
                push!(labels, label)
            end
            if agent_la_type == "I" 
                data = [res[i][agent]["TotalReward"] for i in 1:num_episodes]
                if !("Type1" in keys(plots))
                    p = plot(ExponentialMovingAverage(Winsorise(data, tol), alpha) / (10^4), 
                        xlabel = "Episodes", 
                        title = "Type I",
                        legend = :bottomright, ylabel = raw"Agent Returns [$\times10^{4}$]", label = label, 
                        titlefontsize = 12, 
                        legendfontsize = 8, fg_legend = :transparent, grid = false,
                        fontfamily="Computer Modern")
                    push!(plots, "Type1" => p)
                else
                    plot!(plots["Type1"], ExponentialMovingAverage(Winsorise(data, tol), alpha) / (10^4), 
                        xlabel = "Episodes", 
                    legend = :bottomright, ylabel = raw"Agent Returns [$\times10^{4}$]", label = label, 
                        titlefontsize = 12, 
                    fg_legend = :transparent, grid = false,
                    fontfamily="Computer Modern")
                end
            elseif agent_la_type == "II" 
                data = [res[i][agent]["TotalReward"] for i in 1:num_episodes]
                if !("Type2" in keys(plots))
                    q = plot(ExponentialMovingAverage(Winsorise(data, tol), alpha) / (10^4), 
                        xlabel = "Episodes", 
                        title = "Type II", 
                        legend = :bottomright, ylabel = raw"Agent Returns [$\times10^{4}$]", label = label, 
                        titlefontsize = 12, 
                        legendfontsize = 8, fg_legend = :transparent, grid = false,
                        fontfamily="Computer Modern")
                    push!(plots, "Type2" => q)
                else
                    plot!(plots["Type2"], ExponentialMovingAverage(Winsorise(data, tol), alpha) / (10^4), 
                        xlabel = "Episodes", 
                    legend = :bottomright, ylabel = raw"Agent Returns [$\times10^{4}$]", label = label, 
                        titlefontsize = 12, 
                    legendfontsize = 8, fg_legend = :transparent, grid = false,
                    fontfamily="Computer Modern")
                end
            end
        end
    end
    for key in keys(plots)
#         savefig(plots[key], "../Images/MARL/Returns_" * key * ".pdf")
        display(plots[key])
    end
end

# # read in the type 1 agents data
# t1_b1_s0 = load("../Data/RL/Training/MARL/Results_Type1_Buy_1_Sell_0_alpha0.1_lambda0.003_gamma0.25.jld")["rl_results"];
# t1_b0_s1 = load("../Data/RL/Training/MARL/Results_Type1_Buy_0_Sell_1_alpha0.1_lambda0.003_gamma0.25.jld")["rl_results"];
# t1_b1_s1 = load("../Data/RL/Training/MARL/Results_Type1_Buy_1_Sell_1_alpha0.1_lambda0.003_gamma0.25.jld")["rl_results"];
# t1_b0_s5 = load("../Data/RL/Training/MARL/Results_Type1_Buy_0_Sell_5_alpha0.1_lambda0.003_gamma0.25.jld")["rl_results"];
# t1_b5_s5 = load("../Data/RL/Training/MARL/Results_Type1_Buy_5_Sell_5_alpha0.1_lambda0.003_gamma0.25.jld")["rl_results"];

# # read in the type 2 agents data
# t2_b1_s0 = load("../Data/RL/Training/MARL/Results_Type2_Buy_1_Sell_0_alpha0.1_lambda0.003_gamma0.25_delta_3_6.jld")["rl_results"];
# t2_b1_s1 = load("../Data/RL/Training/MARL/Results_Type2_Buy_1_Sell_1_alpha0.1_lambda0.003_gamma0.25_delta_3_6.jld")["rl_results"];
# t2_b5_s0 = load("../Data/RL/Training/MARL/Results_Type2_Buy_5_Sell_0_alpha0.1_lambda0.003_gamma0.25_delta_3_6.jld")["rl_results"];
# t2_b5_s5 = load("../Data/RL/Training/MARL/Results_Type2_Buy_5_Sell_5_alpha0.1_lambda0.003_gamma0.25_delta_3_6.jld")["rl_results"];

# # read in combined data
# t1_b0_s1_t2_b1_s0 = load("../Data/RL/Training/MARL/Results_Type1_Buy_0_Sell_1_Type2_Buy_1_Sell_0_alpha0.1_lambda0.003_gamma0.25_delta_3_6.jld")["rl_results"];

# marl_data = Dict("I+" => t1_b1_s0, "I-" => t1_b0_s1, "I+_I-" => t1_b1_s1, "II+" => t2_b1_s0,
#     "II+_II-" => t2_b1_s1, "II+_I-" => t1_b0_s1_t2_b1_s0, "5I-" => t1_b0_s5, "5II+" => t2_b5_s0, 
#     "5I+_5I-" => t1_b5_s5, "5II+_5II-" => t2_b5_s5);

# winsorise = true
# winsorise ? tol = 0.01 : tol = 0.0
# moving_average = true
# moving_average ? alpha = 0.1 : alpha = 1.0
# RewardConvergence(marl_data, tol, alpha)
#---------------------------------------------------------------------------------------------------

#----- Phase space reconstruction -----# 

#----- Bin log returns -----# 
function SimpleMovingAverage(logreturns, bins)
    logreturns_binned = Vector{Float64}()
    for i in bins:length(logreturns)
        start = (i-bins)+1
        push!(logreturns_binned, mean(logreturns[start:i]))
    end
    return logreturns_binned
end
#---------------------------------------------------------------------------------------------------

#----- Get delay time parameter -----# 
function GetTauPlotACF(logreturns, lags)
    return_autocor = autocor(logreturns, lags)
    tau = estimate_delay(logreturns, "ac_min", lags)
    println("Tau: " * string(tau))
    plot(return_autocor, xlabel = "Lags", ylabel = "Autocorrelation", legend = false, fontfamily = "Computer Modern")
    return tau
end
#---------------------------------------------------------------------------------------------------

#----- Get linear region to fit linear regression to estimate GP correlation -----# 
function LinearRegion(es_log, cs_log; tol = 0.1)
    diffs = diff(cs_log)
    start_ind = findfirst(x -> x > tol, diffs)
    fin_ind = findfirst(x -> x < tol, diffs[start_ind:end])
    if isnothing(fin_ind)
        fin_ind = length(cs_log)
    end
    return start_ind, fin_ind
end
#---------------------------------------------------------------------------------------------------

#----- Get the slopes of GP dimesnsion vs the box size -----# 
function CorrelationVsBoxSize(data, dims, es_starts, es_stops, es_step, tau, tol, Dmax, plot_slopes)
    
    slopes = Vector{Float64}()
    se_betas = Vector{Float64}()
    
    p = nothing

    for (i, dim) in enumerate(dims)
        # set box size range
        es = ℯ .^ (es_starts[i]:es_step:es_stops[i])
        es_log = log.(es)
        
        # create correlation embeddings
        recon = embed(data, dim, tau)
        cs = correlationsum(recon, es; q = 2)
        cs_log = log.(cs)
        start_ind, fin_ind = LinearRegion(es_log, cs_log; tol = tol)
        if length(cs_log)/2 > length(cs_log[start_ind:fin_ind])
            println("[Warning] Less than a half of the data may be saturated: dim = ", dim)
            println("Fitting to all data, please inspect!")
            start_ind = 1
            fin_ind = length(cs_log)
        end
        lr = linreg(es_log[start_ind:fin_ind], cs_log[start_ind:fin_ind])
        slope = lr[2]
        
        # compute the variance of the slope
        N = length(cs_log)
        sigma_sq = (1/(N-2)) * sum(((lr[1] .+ (lr[2] .* es_log[start_ind:fin_ind])) .- cs_log[start_ind:fin_ind]).^2)
        xbar = mean(es_log[start_ind:fin_ind])
        se_beta = sqrt(sigma_sq / sum((es_log[start_ind:fin_ind] .- xbar).^2))
        
        push!(slopes, slope)
        push!(se_betas, se_beta)
        
        if slope > Dmax
            println("[Warning] Correlation dimension: ", slope, " doesn't have enough data (Dmax = ", Dmax, "), for dim = ",dim)
        end
        
        if plot_slopes
            # make plots
            if i == 1
                p = plot(es_log, cs_log, xlabel = "Box size", ylabel = "Correlation", label = dim, legend = false, linewidth = 1.5, fontfamily = "Computer Modern")
                plot!(p, es_log[start_ind:fin_ind], cs_log[start_ind:fin_ind], linewidth = 1.5, fontfamily = "Computer Modern")
                plot!(es_log, lr[1] .+ (lr[2] .* es_log))
            else
                plot!(p, es_log, cs_log, label = dim, linewidth = 1.5, fontfamily = "Computer Modern")
                plot!(p, es_log[start_ind:fin_ind], cs_log[start_ind:fin_ind], linewidth = 1.5, fontfamily = "Computer Modern")
                plot!(es_log, lr[1] .+ (lr[2] .* es_log))
            end
        end
    end
        
    if plot_slopes
        display(p)
    end
    
    return slopes, se_betas
    
end
#---------------------------------------------------------------------------------------------------

#----- Plot GF (fractal) dimension vs box size -----# 
function PlotEmbeddingDimesion(slopes, se_betas, dims)
    lower = slopes .- 1.96 .* se_betas
    upper = slopes .+ se_betas .* 1.96
    plot(dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", legend = false, 
        marker = "o", markersize = 2, markerstrokewidth = 0, 
        fillrange = (lower, upper), fillalpha=0.2, 
        fontfamily = "Computer Modern")
end
#---------------------------------------------------------------------------------------------------

#----- Smooth embedding dimension using nearest neighbours -----#
# can do ball or square, I will do ball first because it is easiest
function GetNeighbours(data, point, nlast, epsilon)
    neighboursX = Vector{Float64}()
    neighboursY = Vector{Float64}()
    neighboursT = Vector{Int64}()
    neigboursDists = Vector{Float64}()
    for i in 1:size(data)[1]
        # compute dist to all other point
        next_point = data[i,:]
        if next_point.t == point.t # dont compare to itself
            continue
        else
            dist = sqrt((point.x - next_point.x)^2 + (point.y - next_point.y)^2)
            if dist <= epsilon && (point.t - nlast) <= next_point.t && next_point.t < point.t # check if in ball and check if in time bound
                push!(neighboursX, next_point.x)
                push!(neighboursY, next_point.y)
                push!(neighboursT, next_point.t)
                push!(neigboursDists, dist)
            end
        end
    end
    if length(neighboursX) == 0
        neighboursX = point.x
        neighboursY = point.y
        neighboursT = point.t
        neigboursDists = 0
    end
    return neighboursX, neighboursY, neighboursT, neigboursDists
end

function NearestNeighboursSmoothing(data, nlast, epsilon)
    smoothed_vecs = Vector{Vector{Float64}}()
    for i in 1:size(data)[1]
        neighboursX, neighboursY, neighboursT, neigboursDists = GetNeighbours(data, data[i,:], nlast, epsilon)
#         mean(neighboursX[i] * 1/(neigboursDists[i]) for i in 1:length(neighboursT)), mean(neighboursY[i] * 1/(neigboursDists[i]) for i in 1:length(neighboursT))
        if i % 100 == 0
            println("I: ", i," Number of neighbours: ", length(neighboursX), " Average dist: ", mean(neigboursDists))
        end
        push!(smoothed_vecs, [mean(neighboursX), mean(neighboursY)])
    end
    return smoothed_vecs
end
#---------------------------------------------------------------------------------------------------

#----- Plot embedding unsmoothed and smoothed -----#
function PlotEmbedding(embedding, case)
    # plot the embedding
    p = plot([x1 for (x1,x2) in embedding] .* 10^5, [x2 for (x1,x2) in embedding] .* 10^5, 
        title = "Embedding: " * case, 
        legend = false, lw = 1, marker = "o", markersize = 0, alpha = 0.5, color = :dodgerblue3, 
        xlabel = raw"X1 [$\times 10^{5}$]", ylabel = raw"X2 [$\times10^{5}$]", grid = false,
        size = (500,500), fontfamily = "Computer Modern")
  display(p)
end

function PlotEmbedding(embedding, smoothedReturns, indices, case)
    # plot the embedding
    p = plot([x1 for (x1,x2) in embedding] .* 10^5, [x2 for (x1,x2) in embedding] .* 10^5, 
        title = "Embedding: " * case, 
        legend = false, lw = 1, marker = "o", markersize = 0, alpha = 0.5, color = :dodgerblue3, 
        xlabel = raw"X1 [$\times 10^{5}$]", ylabel = raw"X2 [$\times10^{5}$]", grid = false,
        size = (500,500), fontfamily = "Computer Modern")
    if !isnothing(indices)
        plot!([x1 for (x1,x2) in embedding[indices]] .* 10^5, [x2 for (x1,x2) in embedding[indices]] .* 10^5, 
            color = :firebrick, lw = 1)
    end
    
    # plot the returns
    q = plot(smoothedReturns .* 10^5, legend = false, xlabel = "Time", ylabel = raw"Log-returns [$\times10^{5}$]", 
        color = :dodgerblue3, formatter = :plain, 
        title = case, grid = false, size = (500,500), 
        fontfamily = "Computer Modern")
    plot!(indices, smoothedReturns[indices] .* 10^5)
    
    display(p)
    display(q)

    case = join(split(case, ","), "_")
#     savefig(p, "../Images/MARL/EmbbedingExample_" * case * "_no_points_square.pdf")
#     savefig(q, "../Images/MARL/Logreturns_" * case * "_square.pdf")
end
#---------------------------------------------------------------------------------------------------

#----- Perform the state space recon for a single path -----#

#------ Read in data

# RL agent models
# l1lob_data = CSV.File(string("../Data/CoinTossX/MARL/Type2_Buy_1_Sell_0_alpha0.1_lambda0.003_gamma0.25_delta_3_6/L1LOBRLIteration1000.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
# l1lob_data = CSV.File(string("../Data/CoinTossX/MARL/Type1_Buy_1_Sell_1_alpha0.1_lambda0.003_gamma0.25/L1LOBRLIteration1000.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame

# ABM with no RL agents
# l1lob_data = CSV.File(string("../Data/CoinTossX/L1LOBStar.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame

# JSE
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50)
# l1lob_data = CSV.File(string("../Data/JSE/L1LOB.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
# filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, l1lob_data)

# l1lob_data.Date = Date.(l1lob_data.DateTime)
# uniqueDays = unique(l1lob_data.Date)
# uniqueDays = unique(l1lob_data.Date)
# logreturns = map(day -> diff(log.(skipmissing(l1lob_data[searchsorted(l1lob_data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x);

#------set parameters

# preprocessing log-returns
# transient = 500;
# lags = 1:100
# bins = 10

# for pltting GP dimension vs embedding dimension
# Dmax = 2 * log10(length(smoothed_logreturns))
# dims = [1,5,10,20,30,35,40,45,50] 
# es_starts = ones(length(dims)) .* -16 # jse = 10, others = -16
# es_stops = ones(length(dims)) .* -6   # jse = 7.5, others = 6
# es_step = 1 # jse = 0.25, others = 1
# tol = 0.1 # jse = 0.05, others = 0.1
# plot_slopes = true

# for nn smoothing of the embedding
# nlast = 10 # 20 for top one
# epsilon = 1e-3

#------ remove the transient and apply MA

# rtrns_logreturns = logreturns[transient:end];
# smoothed_logreturns = SimpleMovingAverage(rtrns_logreturns, bins);

#------ plot the autocorrelation function and get time delay parameter

# tau = GetTauPlotACF(smoothed_logreturns, lags)

#------ compute the GP correlation dimension and use m to compute the required delay dimension

# tau = tau
# slopes, se_betas = CorrelationVsBoxSize(smoothed_logreturns, dims, es_starts, es_stops, es_step, tau, tol, Dmax, plot_slopes);

# # plots GP dimension vs the embedding dimension 
# PlotEmbeddingDimesion(slopes, se_betas, dims)

# num_dims = 2 # all are 2 and were checked using the above method
# R = embed(smoothed_logreturns, num_dims, tau);

# #------ smooth using nn

# t,(x,y) = collect(1:length(R)), columns(R)
# embedded_dataset = DataFrame(Dict("t"=>t,"x"=>x,"y"=>y))
# smoothed_vecs = NearestNeighboursSmoothing(embedded_dataset, nlast, epsilon);

# # plot the smoothed embedding
# PlotEmbedding(smoothed_vecs, smoothed_logreturns, 6850:7100, "I+,I-")

#---------------------------------------------------------------------------------------------------

#-----  plot all the embeddings vs fractal dimension on the same plot -----#
function GetAllCorrelationVsBoxSize(combinationsAndSettings; plotSlopes = true)
    combinations_res = Dict()
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        # set the settings
        setting = combination_and_setting[2]
        tau = setting[1]
        es_start = setting[2]
        es_stop = setting[3]
        es_step = setting[4]
        tol = setting[5]
        color = setting[6]
        label = setting[7]
        
        # read in the data and generate the log returns
        l1lob_data = nothing
        if types[1] == "jse"
            date = DateTime("2019-07-08")
            startTime = date + Hour(9) + Minute(1)
            endTime = date + Hour(16) + Minute(50)
            l1lob_data = CSV.File(string("../Data/JSE/L1LOB.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
            filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, l1lob_data)
        elseif types[1] == "abm"
            l1lob_data = CSV.File(string("../Data/CoinTossX/L1LOBStar.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        else
            raw_data_dir = CreateCombinations(types, numType1Buyers, numType1Sellers, numType2Buyers, numType2Sellers)
            l1lob_data = CSV.File(string(raw_data_dir * "/L1LOBRLIteration1000.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        end
        l1lob_data.Date = Date.(l1lob_data.DateTime)
        uniqueDays = unique(l1lob_data.Date)
        uniqueDays = unique(l1lob_data.Date)
        simulated_logreturns = map(day -> diff(log.(skipmissing(l1lob_data[searchsorted(l1lob_data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x);
        
        # remove transient and smooth returns 
        transient = 500;
        rtrns_logreturns = simulated_logreturns[transient:end];
        smoothed_logreturns = SimpleMovingAverage(rtrns_logreturns, 10);
        
        # compute the GP correlation dimension and use m to compute the required delay dimension
        Dmax = 2 * log10(length(smoothed_logreturns))
        dims = [1,5,10,20,30,35,40,45,50]
        es_starts = ones(length(dims)) .* es_start
        es_stops = ones(length(dims)) .* es_stop
        
        slopes, se_betas = CorrelationVsBoxSize(smoothed_logreturns, dims, es_starts, es_stops, es_step, tau, tol, Dmax, plotSlopes);
        lower = slopes .- 1.96 .* se_betas
        upper = slopes .+ 1.96 .* se_betas  
        
        push!(combinations_res, combination => Dict("slopes" => slopes, "se_betas" => se_betas))
    end
    return combinations_res
end

function PlotAllCorrelationVsBoxSize(combinationsRes, combinationsAndSettings)
    p = nothing
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        # set the settings
        setting = combination_and_setting[2]
        tau = setting[1]
        es_start = setting[2]
        es_stop = setting[3]
        es_step = setting[4]
        tol = setting[5]
        color = setting[6]
        label = setting[7]
        buy_sell_mixed = setting[8]

        # set the settings
        Dmax = 2 * log10(length(smoothed_logreturns))
        dims = [1,5,10,20,30,35,40,45,50]
        es_starts = ones(length(dims)) .* es_start
        es_stops = ones(length(dims)) .* es_stop
        
        # get slopes and intervals
        slopes = combinationsRes[combination]["slopes"]
        se_errors = combinationsRes[combination]["se_betas"]
        lower = slopes .- 1.96 .* se_betas
        upper = slopes .+ 1.96 .* se_betas
        
        if buy_sell_mixed == "buy"
            markershape = :utriangle
        elseif buy_sell_mixed == "sell"
            markershape = :dtriangle
        else
            markershape = :circle
        end

        # plot slopes
        if i == 1
            if types[1] == "jse" || types[1] == "abm"
                p = plot(dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", 
                    color = color, linewidth = 3,
                label = label, legend = :topleft, fillrange = (lower, upper), fillalpha=0.3,
                marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape,
                      markerstrokecolor = color,
                fg_legend = :transparent, grid = false, 
                size=(800,600), fontfamily = "Computer Modern")
            else
                p = plot(dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", color = color,
                label = label, legend = :topleft, linewidth = 2,
                marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape,
                     markerstrokecolor = color,
                fg_legend = :transparent, grid = false, 
                size=(800,600), fontfamily = "Computer Modern")
            end
            #   fillrange = (lower, upper), fillalpha=0.2, 
            if types[1] != "jse"
                plot!(p, dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", 
                    color = color, linewidth = 2,
                    legend = false, inset = (1, bbox(0.6, 0.2, 0.4, 0.35)), subplot = 2,
                    marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape,
                      markerstrokecolor = color,
                    fg_legend = :transparent, grid = false, 
                    fontfamily = "Computer Modern")
                    # inset = (1, bbox(0.2, 0.5, 0.5, 0.4)), (1, bbox(0.55, 0.2, 0.4, 0.35))
            end
        else
            if types[1] == "jse" || types[1] == "abm"
                plot!(p, dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension",
                    color = color, linewidth = 3,
                label = label, legend = :topleft, fillrange = (lower, upper), fillalpha=0.3,
                marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape,
                     markerstrokecolor = color,
                fg_legend = :transparent, grid = false, 
                fontfamily = "Computer Modern")
            else
                plot!(p, dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", color = color, 
                label = label, legend = :topleft, linewidth = 2,
                marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape, 
                    markerstrokecolor = color,
                fg_legend = :transparent, grid = false, 
                fontfamily = "Computer Modern")
            end
            
            if types[1] != "jse"
                plot!(p[2], dims, slopes, xlabel = "Embedding dimension", ylabel = "Fractal dimension", 
                    color = color, linewidth = 2,
                    legend = false, #yaxis = :log, #fillrange = (lower, upper), fillalpha=0.2,
                    marker = "o", markersize = 4, markerstrokewidth = 0, markershape = markershape,
                     markerstrokecolor = color,
                    fg_legend = :transparent, grid = false, 
                    fontfamily = "Computer Modern")
            end
        end
    end
    # savefig(p, "../Images/MARL/EmbeddingVsFractal.pdf")
    display(p)
end

function AvgSlopesVsNumAgents(combinationsRes, combinationsAndSettings, window)
    num_agent_slopes = Dict{Int64, Vector{Float64}}()
    
    avg_abm_slope = mean(combinationsRes[(["abm"], 0, 0, 0, 0)]["slopes"][(end-(window-1)):end])
    
    println("Average ABM slope => ", avg_abm_slope)
    println()
    println("Agent slope differences: ")
    
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        
        # set the settings
        setting = combination_and_setting[2]
        num_agents = setting[9]
        label = setting[7]
        
        if num_agents == 0
            continue
        end
        
        # get slopes and intervals
        slopes = combinationsRes[combination]["slopes"]
        if num_agents in keys(num_agent_slopes)
            append!(num_agent_slopes[num_agents], (slopes[(end-(window-1)):end]) .- (avg_abm_slope))
        else
            push!(num_agent_slopes, num_agents => (slopes[(end-(window-1)):end]) .- (avg_abm_slope))
        end
        
        println(label, " => ", mean(slopes[(end-(window-1)):end]) .- (avg_abm_slope))
        
    end
    println()
    println("Slope vs Number of agents: ")
    for key in keys(num_agent_slopes)
        println(key, " => ", mean(num_agent_slopes[key]))
    end
end

# combinations_and_settings = [
#     (([1], 1, 0, 0, 0), (6, -15, -6, 1, 0.1, mycolorscheme[2], "I+", "buy", 1)),
#     (([1], 0, 1, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[3], "I-", "sell", 1)),
#     (([1], 1, 1, 0, 0), (6, -16, -6, 1, 0.1, mycolorscheme[5], "I+,I-", "mixed", 2)),
#     (([2], 0, 0, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[4], "II+", "buy", 1)),
#     (([2], 0, 0, 1, 1), (10, -16, -6, 1, 0.1, mycolorscheme[6], "II+,II-", "mixed", 2)),
#     (([1,2], 0, 1, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[7], "II+,I-", "mixed", 2)),
#     (([1], 0, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[8], "5I-", "sell", 5)),
#     (([2], 0, 0, 5, 0), (10, -16, -6, 1, 0.1, mycolorscheme[9], "5II+", "buy", 5)),
#     (([1], 5, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[10], "5I+,5I-", "mixed", 10)),
#     (([2], 0, 0, 5, 5), (10, -16, -6, 1, 0.1, mycolorscheme[11], "5II+,5II-", "mixed", 10)),
#     ((["abm"], 0, 0, 0, 0), (10, -16, -6, 1, 0.1, :black, "ABM", "none", 0)),
#     ((["jse"], 0, 0, 0, 0), (10, -10, -7, 0.25, 0.05, mycolorscheme[1], "JSE", "none", 0)),
# ];

# combinations_res = GetAllCorrelationVsBoxSize(combinations_and_settings);

# mycolorscheme = [
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.76,0.97,3))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.3,0.5,4))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.04,0.2,4))]
# ]
# PlotAllCorrelationVsBoxSize(combinations_res, combinations_and_settings)

# AvgSlopesVsNumAgents(combinations_res, combinations_and_settings, 5)
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

#----- Trade sign autocorrelation -----#
function RLTradeSignAutocorrelation(combinationsAndSettings::Vector, format::String = "pdf")
    autoCorrPlot = nothing
    tradeSigns = nothing
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        # set the settings
        setting = combination_and_setting[2]
        tau = setting[1]
        es_start = setting[2]
        es_stop = setting[3]
        es_step = setting[4]
        tol = setting[5]
        color = setting[6]
        label = setting[7]
        buy_sell_mixed = setting[8]

        raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
        if 1 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
        end
        if 2 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
        end
        raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
        if 2 in types
            raw_data_dir = raw_data_dir * "_delta_3_6"
        end
        
        data = CSV.File(string(raw_data_dir * "/L1LOBRLIteration1000.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        data.Date = Date.(data.DateTime)
        uniqueDays = unique(data.Date)
        tradeSigns = data[findall(x -> x == :Market, data.Type), :Side]
        lag = length(tradeSigns) - 1
        autoCorr = autocor(tradeSigns, 1:lag; demean = false)
        if i == 1
            autoCorrPlot = plot(autoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = (0.35,0.9), label = label, title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), grid = false, fontfamily="Computer Modern")
            plot!(autoCorrPlot, autoCorr, xscale = :log10, inset = (1, bbox(0.58, 0.1, 0.4, 0.4)), legend = false, subplot = 2, xlabel = "Lag", guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, ylabel = "Autocorrelation", linecolor = color, title = "Log-scale order-flow autocorrelation", titlefontsize = 7, ylim = (-0.1, 0.5), grid = false, fontfamily="Computer Modern")
        else
            plot!(autoCorrPlot, autoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = (0.35,0.9), label = label, title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), grid = false, fontfamily="Computer Modern")
            plot!(autoCorrPlot[2], autoCorr, xscale = :log10, legend = false, xlabel = "Lag", guidefontsize = 7, tickfontsize = 5, xrotation = 30, yrotation = 30, ylabel = "Autocorrelation", linecolor = color, title = "Log-scale order-flow autocorrelation", titlefontsize = 7, ylim = (-0.1, 0.5  ))
        end
    end
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(tradeSigns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(tradeSigns))], seriestype = :hline, line = (:dash, :black, 2), label = "")
    savefig(autoCorrPlot, "../Images/MARL/RLTradeSignAutocorrelation." * format)
    println("RL agents trade-sign autocorrelation complete")
end

# mycolorscheme = [
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.76,0.97,3))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.3,0.5,4))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.04,0.2,4))]
# ]
# combinations_and_settings = [
#     (([1], 1, 0, 0, 0), (6, -15, -6, 1, 0.1, mycolorscheme[2], "I+", "buy", 1)),
#     (([1], 0, 1, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[3], "I-", "sell", 1)),
#     (([1], 1, 1, 0, 0), (6, -16, -6, 1, 0.1, mycolorscheme[5], "I+,I-", "mixed", 2)),
#     (([2], 0, 0, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[4], "II+", "buy", 1)),
#     (([2], 0, 0, 1, 1), (10, -16, -6, 1, 0.1, mycolorscheme[6], "II+,II-", "mixed", 2)),
#     (([1,2], 0, 1, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[7], "II+,I-", "mixed", 2)),
#     (([1], 0, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[8], "5I-", "sell", 5)),
#     (([2], 0, 0, 5, 0), (10, -16, -6, 1, 0.1, mycolorscheme[9], "5II+", "buy", 5)),
#     (([1], 5, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[10], "5I+,5I-", "mixed", 10)),
#     (([2], 0, 0, 5, 5), (10, -16, -6, 1, 0.1, mycolorscheme[11], "5II+,5II-", "mixed", 10)),
# #     ((["abm"], 0, 0, 0, 0), (10, -16, -6, 1, 0.1, :black, "ABM", "none", 0)),
# ];

# RLTradeSignAutocorrelation(combinations_and_settings);
#---------------------------------------------------------------------------------------------------

#----- Absolute log return autocorrelation -----#
function RLAbsLogReturnAutocorrelation(combinationsAndSettings::Vector, format::String = "pdf")
    autoCorrPlot = nothing
    logreturns = nothing
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        # set the settings
        setting = combination_and_setting[2]
        tau = setting[1]
        es_start = setting[2]
        es_stop = setting[3]
        es_step = setting[4]
        tol = setting[5]
        color = setting[6]
        label = setting[7]
        buy_sell_mixed = setting[8]

        raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
        if 1 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
        end
        if 2 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
        end
        raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
        if 2 in types
            raw_data_dir = raw_data_dir * "_delta_3_6"
        end
        
        data = CSV.File(string(raw_data_dir * "/L1LOBRLIteration1000.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        data.Date = Date.(data.DateTime)
        uniqueDays = unique(data.Date)
        logreturns = map(day -> diff(log.(skipmissing(data[searchsorted(data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x)
        lag = length(logreturns) - 1
        absAutoCorr = autocor(abs.(logreturns), 1:lag; demean = false)
        if i == 1
            autoCorrPlot = plot(absAutoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = label, title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), formatter = :plain, grid = false, fontfamily="Computer Modern")
        else
            plot!(autoCorrPlot, absAutoCorr, seriestype = :scatter, linecolor = :black, marker = (color, stroke(color), 3), legend = :topleft, label = label, title = "", xlabel = "Lag", ylabel = "Autocorrelation", fg_legend = :transparent, ylim = (-0.1, 0.8), formatter = :plain, grid = false, fontfamily="Computer Modern")
        end
    end
    plot!(autoCorrPlot, [quantile(Normal(), (1 + 0.95) / 2) / sqrt(length(logreturns)), quantile(Normal(), (1 - 0.95) / 2) / sqrt(length(logreturns))], seriestype = :hline, line = (:dash, :black, 2), label = "")
    savefig(autoCorrPlot, "../Images/MARL/RLAbsLogReturnAutocorrelation." * format)
    println("RL agents absolute log return autocorrelation complete")
end

# mycolorscheme = [
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.76,0.97,3))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.3,0.5,4))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.04,0.2,4))]
# ]
# combinations_and_settings = [
#     (([1], 1, 0, 0, 0), (6, -15, -6, 1, 0.1, mycolorscheme[2], "I+", "buy", 1)),
#     (([1], 0, 1, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[3], "I-", "sell", 1)),
#     (([1], 1, 1, 0, 0), (6, -16, -6, 1, 0.1, mycolorscheme[5], "I+,I-", "mixed", 2)),
#     (([2], 0, 0, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[4], "II+", "buy", 1)),
#     (([2], 0, 0, 1, 1), (10, -16, -6, 1, 0.1, mycolorscheme[6], "II+,II-", "mixed", 2)),
#     (([1,2], 0, 1, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[7], "II+,I-", "mixed", 2)),
#     (([1], 0, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[8], "5I-", "sell", 5)),
#     (([2], 0, 0, 5, 0), (10, -16, -6, 1, 0.1, mycolorscheme[9], "5II+", "buy", 5)),
#     (([1], 5, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[10], "5I+,5I-", "mixed", 10)),
#     (([2], 0, 0, 5, 5), (10, -16, -6, 1, 0.1, mycolorscheme[11], "5II+,5II-", "mixed", 10)),
# #     ((["abm"], 0, 0, 0, 0), (10, -16, -6, 1, 0.1, :black, "ABM", "none", 0)),
# ];
# RLAbsLogReturnAutocorrelation(combinations_and_settings);

#---------------------------------------------------------------------------------------------------

#----- Price impact -----#
function RLPriceImpact(combinationsAndSettings::Vector, format::String = "pdf")
    println("Computing RL agents price impact")
    println("Reading in data...")
    priceImpactBuy = nothing
    priceImpactSell = nothing
    for (i, combination_and_setting) in enumerate(combinationsAndSettings)
        
        # get the number of buyers and sellers
        combination = combination_and_setting[1]
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        # set the settings
        setting = combination_and_setting[2]
        tau = setting[1]
        es_start = setting[2]
        es_stop = setting[3]
        es_step = setting[4]
        tol = setting[5]
        color = setting[6]
        label = setting[7]
        buy_sell_mixed = setting[8]

        raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
        if 1 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
        end
        if 2 in types
            raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
        end
        raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
        if 2 in types
            raw_data_dir = raw_data_dir * "_delta_3_6"
        end
        
        data = CSV.File(string(raw_data_dir * "/L1LOBRLIteration1000.csv"), drop = [:MicroPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
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
        if i == 1
            priceImpactBuy = plot(ω[2:(end-3), 1], Δp[2:(end-3), 1], scale = :log10, markershape = [:utriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = label, legend = :topleft, fg_legend = :transparent, title = "Buyer initiated", grid = false, fontfamily="Computer Modern")
            priceImpactSell = plot(ω[2:(end-3), 2], Δp[2:(end-3), 2], scale = :log10,  markershape = [:dtriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = label, legend = :topleft, fg_legend = :transparent, title = "Seller initiated", grid = false, fontfamily="Computer Modern")
        else
            plot!(priceImpactBuy, ω[2:(end-3), 1], Δp[2:(end-3), 1], scale = :log10, markershape = [:utriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = label, legend = :topleft, fg_legend = :transparent, title = "Buyer initiated", grid = false, fontfamily="Computer Modern")
            plot!(priceImpactSell, ω[2:(end-3), 2], Δp[2:(end-3), 2], scale = :log10, markershape = [:dtriangle], markercolor = [color], markerstrokecolor = [color], markersize = 3, linecolor = [color], xlabel = "ω*", ylabel = "Δp*", label = label, legend = :topleft, fg_legend = :transparent, title = "Seller initiated", grid = false, fontfamily="Computer Modern")
        end
    end
    savefig(priceImpactBuy, string("../Images/MARL/RLPriceImpactBuyerInitiated.", format))
    savefig(priceImpactSell, string("../Images/MARL/RLPriceImpactSellerInitiated.", format))
    println("RL agents price impact complete")
end

# mycolorscheme = [
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.76,0.97,3))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.3,0.5,4))];
#     [get(colorschemes[:nipy_spectral], i) for i in reverse(range(0.04,0.2,4))]
# ]
# combinations_and_settings = [
#     (([1], 1, 0, 0, 0), (6, -15, -6, 1, 0.1, mycolorscheme[2], "I+", "buy", 1)),
#     (([1], 0, 1, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[3], "I-", "sell", 1)),
#     (([1], 1, 1, 0, 0), (6, -16, -6, 1, 0.1, mycolorscheme[5], "I+,I-", "mixed", 2)),
#     (([2], 0, 0, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[4], "II+", "buy", 1)),
#     (([2], 0, 0, 1, 1), (10, -16, -6, 1, 0.1, mycolorscheme[6], "II+,II-", "mixed", 2)),
#     (([1,2], 0, 1, 1, 0), (10, -16, -6, 1, 0.1, mycolorscheme[7], "II+,I-", "mixed", 2)),
#     (([1], 0, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[8], "5I-", "sell", 5)),
#     (([2], 0, 0, 5, 0), (10, -16, -6, 1, 0.1, mycolorscheme[9], "5II+", "buy", 5)),
#     (([1], 5, 5, 0, 0), (10, -16, -6, 1, 0.1, mycolorscheme[10], "5I+,5I-", "mixed", 10)),
#     (([2], 0, 0, 5, 5), (10, -16, -6, 1, 0.1, mycolorscheme[11], "5II+,5II-", "mixed", 10)),
# #     ((["abm"], 0, 0, 0, 0), (10, -16, -6, 1, 0.1, :black, "ABM", "none", 0)),
# ];

# RLPriceImpact(combinations_and_settings)
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

#----- Model distance function plots -----# 

# Computes the Model distance function
function MDF(W, simulatedMoments, abmMoments)
    abmMoments_mat = [abmMoments.μ abmMoments.σ abmMoments.ks abmMoments.hurst abmMoments.gph abmMoments.adf abmMoments.garch abmMoments.hill]
    simulatedMoments_mat = [simulatedMoments.μ simulatedMoments.σ simulatedMoments.ks simulatedMoments.hurst simulatedMoments.gph simulatedMoments.adf simulatedMoments.garch simulatedMoments.hill]
    errors = simulatedMoments_mat .- abmMoments_mat
    return (errors * W * transpose(errors))[1], errors
end

# Compute the model distance function for a specific buyer seller combination RL agent
function SingleCombinationMDF(W, abmMoments::Moments, empiricalLogreturns::Vector{Float64}, empiricalMoments::Moments, types::Vector{Int64}, numType1Buyers::Int64, numType1Sellers::Int64, numType2Buyers::Int64, numType2Sellers::Int64, window::Int64)
    # create folder path to raw data
    raw_data_dir = path_to_files * "/Data/CoinTossX/MARL/"
    if 1 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 1, types)][1]) * "_Buy_" * string(numType1Buyers) * "_Sell_" * string(numType1Sellers) * "_"
    end
    if 2 in types
        raw_data_dir = raw_data_dir * "Type" * string(types[findall(x -> x == 2, types)][1]) * "_Buy_" * string(numType2Buyers) * "_Sell_" * string(numType2Sellers) * "_"
    end
    raw_data_dir = raw_data_dir * "alpha0.1_lambda0.003_gamma0.25"
    if 2 in types
        raw_data_dir = raw_data_dir * "_delta_3_6"
    end
    
    simulated_moments_final = nothing
    errors_final = nothing
    mdf_values = Vector{Float64}()
    for i in 0:window
        
        iteration = 1000 - (i * 10)
        
        # read in the data and generate the log returns
        l1lob_data = CSV.File(string(raw_data_dir * "/L1LOBRLIteration" * string(iteration) * ".csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
        l1lob_data.Date = Date.(l1lob_data.DateTime)
        uniqueDays = unique(l1lob_data.Date)
        uniqueDays = unique(l1lob_data.Date)
        simulated_logreturns = map(day -> diff(log.(skipmissing(l1lob_data[searchsorted(l1lob_data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x);

        # compute the moments 
        simulated_moments = Moments(simulated_logreturns, empiricalLogreturns)
        mdf_value, errors = MDF(W, simulated_moments, abmMoments)
        push!(mdf_values, mdf_value)
        if iteration == 1000
            simulated_moments_final = simulated_moments
            errors_final = errors
        end
    end
    return simulated_moments_final, errors_final, mdf_values
end

# Compute the model distance function for a all buyer seller combinations
function AllCombinationsMDF(W, abmMoments::Moments, empiricalLogreturns::Vector{Float64}, empiricalMoments::Moments, combinations::Vector{Tuple{Vector{Int64}, Int64, Int64, Int64, Int64}}, window::Int64)

    results = Dict()
    
    for combination in combinations
        types = combination[1]
        numType1Buyers = combination[2]
        numType1Sellers = combination[3]
        numType2Buyers = combination[4]
        numType2Sellers = combination[5] 
        
        simulated_moments_final, errors_final, mdf_values = SingleCombinationMDF(W, abmMoments, empiricalLogreturns, empiricalMoments, types, numType1Buyers, numType1Sellers, numType2Buyers, numType2Sellers, window)

        push!(results, combination => Dict("simulated_moments_final" => simulated_moments_final, "errors_final" => errors_final, "mdf_values" => mdf_values))
        
    end
    
    return results
    
end

function MDFTable(results::Dict)
    for key in keys(results)
        mean_mdf = mean(results[key]["mdf_values"])
        std_mdf = std(results[key]["mdf_values"])
        println(key, " => ", mean_mdf, " +- ", 1.96 * std_mdf)
    end
end

# # read in jse data and get the log returns 
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50) 

# jse_l1lob_data = CSV.File(string("../Data/JSE/L1LOB.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
# filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, jse_l1lob_data)

# jse_l1lob_data.Date = Date.(jse_l1lob_data.DateTime)
# uniqueDays = unique(jse_l1lob_data.Date)
# uniqueDays = unique(jse_l1lob_data.Date)
# jse_logreturns = map(day -> diff(log.(skipmissing(jse_l1lob_data[searchsorted(jse_l1lob_data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x)
# jse_moments = Moments(jse_logreturns, jse_logreturns);

# # read in abm data and compute the moments
# abm_l1lob_data = CSV.File(string("../Data/CoinTossX/L1LOBStar.csv"), drop = [:MidPrice, :Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
# abm_l1lob_data.Date = Date.(abm_l1lob_data.DateTime)
# uniqueDays = unique(abm_l1lob_data.Date)
# uniqueDays = unique(abm_l1lob_data.Date)
# abm_logreturns = map(day -> diff(log.(skipmissing(abm_l1lob_data[searchsorted(abm_l1lob_data.Date, day), :MicroPrice]))), uniqueDays) |> x -> reduce(vcat, x);
# abm_moments = Moments(abm_logreturns, jse_logreturns);
        
# W = load("../Data/Calibration/W.jld")["W"];

# combinations = [
#     ([1], 1, 0, 0, 0),
#     ([1], 0, 1, 0, 0), 
#     ([1], 1, 1, 0, 0),
#     ([2], 0, 0, 1, 0), 
#     ([2], 0, 0, 1, 1), 
#     ([1,2], 0, 1, 1, 0), 
#     ([1], 0, 5, 0, 0), 
#     ([2], 0, 0, 5, 0), 
#     ([1], 5, 5, 0, 0),
#     ([2], 0, 0, 5, 5)
# ];
# window = 10;

# results = AllCombinationsMDF(W, abm_moments, jse_logreturns, jse_moments, combinations, window)

# MDFTable(results)

#---------------------------------------------------------------------------------------------------

#----- Moment comparison plots -----# 

function AbsoluteMomentBarPlot(results::Dict, abmMoments::Moments, jseMoments::Moments, combinations::Vector, combinationsAndLegend::Vector)

    # radar 1: absolute moments
    labels = [raw"$\mu$", raw"$\sigma$", "KS", "Hurst", "GPH", "ADF", "GARCH", "Hill"]
        
    # normalise
    moment_mat = fill(0.0, (length(combinations), length(labels)))
    abm_index = 0
    indices = Dict()
    for (i, (combination, case)) in enumerate(combinations)
        if case == "ABM"
            abm_index = i
            moments = abmMoments
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        elseif case == "JSE"
            moments = jseMoments
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        else
            moments = results[combination]["simulated_moments_final"]
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        end
        if combination in first.(combinationsAndLegend)
            push!(indices, combination => i)
        end
    end
    
    means = mean(moment_mat, dims = 1)
    sdevs = std(moment_mat, dims = 1)
    
    moments_norm = (moment_mat .- means) ./ sdevs 
    
    p = nothing
    cases = Vector{String}()
    moments_vec = Vector{Float64}()
    colors = Vector()
    for (i, (combination, legend, color)) in enumerate(combinationsAndLegend)
        
        if legend == "ABM"
            moments = abmMoments
            append!(moments_vec, moments_norm[indices[combination],:])
        elseif legend == "JSE"
            moments = jseMoments
            append!(moments_vec, moments_norm[indices[combination],:])
        else
            moments = results[combination]["simulated_moments_final"]
            append!(moments_vec, moments_norm[indices[combination],:])
        end
        push!(cases, legend)
        push!(colors, color)
    end
    # reshape(colors, (1,length(colors)))
    p = groupedbar(repeat(labels, outer = length(cases)), moments_vec, 
        group = repeat(cases, inner = length(labels)), legend = :topleft, fg_legend = :transparent, 
        ylabel = "Absolute normalised moments", grid = false, color = reshape(colors, (1, length(colors))), 
        linecolor = :match,
        font="Computer Modern")

    display(p)
#     savefig(p, "../Images/MARL/AbsoluteMomentBarPlot.pdf")
end

function RelativeMomentBarPlot(results::Dict, abmMoments::Moments, jseMoments::Moments, combinations::Vector, combinationsAndLegend::Vector)

    # radar 1: absolute moments
    labels = [raw"$\mu$", raw"$\sigma$", "KS", "Hurst", "GPH", "ADF", "GARCH", "Hill"]
        
    # normalise
    moment_mat = fill(0.0, (length(combinations), length(labels)))
    abm_index = 0
    indices = Dict()
    for (i, (combination, case)) in enumerate(combinations)
        if case == "ABM"
            abm_index = i
            moments = abmMoments
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        elseif case == "JSE"
            moments = jseMoments
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        else
            moments = results[combination]["simulated_moments_final"]
            moment_mat[i,:] = [moments.μ moments.σ moments.ks moments.hurst moments.gph moments.adf moments.garch moments.hill]
        end
        if combination in first.(combinationsAndLegend)
            push!(indices, combination => i)
        end
    end
    
    means = mean(moment_mat, dims = 1)
    sdevs = std(moment_mat, dims = 1)
    
    moments_norm = (moment_mat .- means) ./ sdevs 
    
    p = nothing
    cases = Vector{String}()
    moments_vec = Vector{Float64}()
    colors = Vector()
    for (i, (combination, legend, color)) in enumerate(combinationsAndLegend)
        
        if legend != "JSE" && legend != "ABM" # only consider the cases 
            moments = results[combination]["simulated_moments_final"]
            append!(moments_vec, abs.(moments_norm[indices[combination],:] .- moments_norm[abm_index,:]))
            push!(cases, legend)
            push!(colors, color)
        end

    end

    # reshape(colors, (1,length(colors)))
    p = groupedbar(repeat(labels, outer = length(cases)), moments_vec, 
        group = repeat(cases, inner = length(labels)), legend = :topleft, fg_legend = :transparent, 
        ylabel = "Relative normalised moments", grid = false, color = reshape(colors, (1, length(colors))), 
        linecolor = :match,
        font="Computer Modern")

    display(p)
#     savefig(p, "../Images/MARL/RelativeMomentBarPlot.pdf")
end

mycolorpalette = palette(:default)
combinations_and_legends = [
    ((["abm"], 0, 0, 0, 0), "ABM", mycolorpalette[1]),
    (([2], 0, 0, 1, 0), "II+", mycolorpalette[2]),
    (([1], 1, 1, 0, 0), "I+,I-", mycolorpalette[3]),
    ((["jse"], 0, 0, 0, 0), "JSE", mycolorpalette[4])
];
combinations = [
    (([1], 1, 0, 0, 0), "I+"),
    (([1], 0, 1, 0, 0), "I-"),
    (([1], 1, 1, 0, 0), "I+,I-"),
    (([2], 0, 0, 1, 0), "II+"),
    (([2], 0, 0, 1, 1), "II+,II-"),
    (([1,2], 0, 1, 1, 0), "II+,I-"),
    (([1], 0, 5, 0, 0), "5I-"),
    (([2], 0, 0, 5, 0), "5II+"),
    (([1], 5, 5, 0, 0), "5I+,5I-"),
    (([2], 0, 0, 5, 5), "5II+,5II-"),
    ((["abm"], 0, 0, 0, 0), "ABM"),
    ((["jse"], 0, 0, 0, 0), "JSE")
];
# RelativeMomentBarPlot(results, jse_moments, abm_moments, combinations, combinations_and_legends)
# AbsoluteMomentBarPlot(results, jse_moments, abm_moments, combinations, combinations_and_legends)
#---------------------------------------------------------------------------------------------------