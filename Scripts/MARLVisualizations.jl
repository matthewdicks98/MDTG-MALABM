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
using DataFrames, CSV, Plots, Statistics, DataStructures, JLD, Plots.PlotMeasures, LaTeXStrings

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/Moments.jl"); 

#----- Reward convergence for all buyer seller RL agents -----# 
function RewardConvergence(rl_results::Dict)

end
#---------------------------------------------------------------------------------------------------

#----- Clean all raw message data into L1LOB format -----# 
function CleanAllAgentData()

end
#---------------------------------------------------------------------------------------------------

#----- Model distance function plots -----# 

#----- Computes the Model distance function -----# 
function MDF(empiricalMoments, simulatedMoments)

end
#---------------------------------------------------------------------------------------------------

#----- Compute the model distance function for a specific buyer seller combination RL agent-----# 
function SingleCombinationMDF(numBuyers::Int64, numSellers::Int64)

end
#---------------------------------------------------------------------------------------------------

#----- Compute the model distance function for a all buyer seller combinations -----# 
function AllCombinationsMDF(combinations::Vector{Tuple{Int64, Int64}})

end
#---------------------------------------------------------------------------------------------------

#----- Plot MDF heatmap for all buyer and seller combinations using last iteration -----# 
function PlotHeatmapAllCombinationsMDF(combinations::Vector{Tuple{Int64, Int64}}, allMDFs::Dict{Tuple{Int64, Int64}, Float64})

end
#---------------------------------------------------------------------------------------------------

#----- Plot the MDF over time for all buyer and seller combinations -----# 
function PlotDynamicsAllCombinationsMDF(combinations::Vector{Tuple{Int64, Int64}}, allMDFs::Dict{Tuple{Int64, Int64}, Float64})

end
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------

#----- Phase space reconstruction (for 1 combination) -----# 

#----- Bin log returns -----# 
function Bin(logreturns::Vector{Float64}, bins::Int64)
    logreturns_binned = Vector{Float64}()
    for i in bins:length(logreturns)
        start = (i-bins)+1
        push!(logreturns_binned, mean(logreturns[start:i]))
    end
    return logreturns_binned
end
#---------------------------------------------------------------------------------------------------

#----- Bin and truncate log returns -----# 
function TruncateBin(logreturns::Vector{Float64}, bins::Int64)

end
#---------------------------------------------------------------------------------------------------

#----- Plot Binned and truncated log returns -----#
function PlotTruncatedBinnedReturns(logreturns::Vector{Float64})

end
#---------------------------------------------------------------------------------------------------

#----- Plot autocorrelation and estimate time delay parameter tau -----#
function EstimateTimeDelay(logreturns::Vector{Float64})

end

function PlotAutocorrelation(logreturns::Vector{Float64})

end
#---------------------------------------------------------------------------------------------------

#----- Compute the correlation dimension and plot slope convergence (sanity check) -----#
function PlotCorrelationDimSlopes(data, dims, es_starts, es_stops, es_step, tau)

end
#---------------------------------------------------------------------------------------------------

#----- Plot the correlation dimension vs embedding size -----#
function PlotCorrelationDimVSEmbeddingDim(slopes::Vector{Float64})

end
#---------------------------------------------------------------------------------------------------

#----- Smooth embedding dimension using nearest neighbours -----#
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
function PlotEmbeddings(unSmoothedData, smoothedData)

end
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------