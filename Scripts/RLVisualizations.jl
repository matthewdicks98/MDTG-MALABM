#=
RLVisualisations:
- Julia version: 1.7.1
- Authors: Matthew Dicks, Tim Gebbie
- Function: Visualise the data generated from training the RL agents
- Structure:
    1. Epsion decay
    2. Returns, states, trades, Q-matrix, and policy convergence
    3. State-action convergence
    4. Visualising the policy
    5. Learning dynamics (average actions taken over time)
    6. Volume trajectory
    7. Estimated stylised fact moments over training period
- Examples:
    1. EpsilonDecay(steps, sizes)
    2. PlotRLConvergenceResults(actions)
    3. StateActionConvergence(l, numT, I, B, W, A, V, actions)
    4. PolicyVisualization(l[1000]["Q"], numT, I, B, W, A, V, actions)
    5. PlotAverageActionsOverTime(l, numT, I, B, W, A, V, actions)
    6. PlotVolumeTragectory(l, numT, I, B, W, A, V, actions)
    7. PlotStylisedFactDynamics(empericalLogReturns, moments, abm_moments_df)
=#
ENV["JULIA_COPY_STACKS"]=1
using DataFrames, CSV, Plots, Statistics, DataStructures, JLD, Plots.PlotMeasures, LaTeXStrings

path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/Moments.jl"); include(path_to_files * "DataCleaning/CoinTossX.jl")

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

#---------------------------------------------------------------------- Visualizations ----------------------------------------------------------------------#

#----- Plots the decaying epsilon -----# 
function EpsilonDecay(steps::Vector{Int64}, sizes::Vector{Float64})
    xs = Vector()
    x = xinit = 1
    steps = steps
    sizes = sizes
    j = 1
    for i in 1:sum(steps)
        x -= (xinit - (1 - sizes[j]) * xinit)/steps[j]  
        if i == sum(steps[1:j])
            j += 1
        end  
        push!(xs, max(x,0))
    end
    p = plot(xs, xlabel = "Episodes", ylabel = L"\epsilon", color = :black, legend = false, fontfamily="Computer Modern")
    savefig(p, path_to_files * "/Images/RL/EpsilonDecay.pdf")
end
# steps = [200, 400, 150, 250] 
# sizes = [0.1, 0.8, 0.09, 0]
# EpsilonDecay(steps, sizes)
#---------------------------------------------------------------------------------------------------

#----- Plot the RL training results -----# 
function PlotRLConvergenceResults(actionsMap::Dict)

    # read in data into a dict
    data = Dict()
    Vs = [200, 100, 50]
    num_states = [5,10]
    for V in Vs
        for num_state in num_states
            @time d = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state) * "_430.jld")["rl_results"]
            push!(data, "Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state) => d)
        end
    end

    # convergence of rewards
    reward_plots = []
    Vs = [200, 100, 50]
    for V in Vs
        rewards5 = Vector{Float64}()
        rewards10 = Vector{Float64}()
        l5 = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S5"]
        l10 = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S10"]
        n = length(l5)
        for i in 1:n
            push!(rewards5, l5[i]["TotalReward"])
            push!(rewards10, l10[i]["TotalReward"])
        end

        V == 100 ? ylab = "Inventory Normalized Total Profit" : ylab = ""

        if V == 200 # used to add winsorization

            # winsorize because the reward convergence gets distorted
            rewards5 = rewards5[findall(x -> x < quantile(rewards5, 0.99) && x > quantile(rewards5, 0.01), rewards5)]
            rewards10 = rewards10[findall(x -> x < quantile(rewards10, 0.99) && x > quantile(rewards10, 0.01), rewards10)]

            p = plot(rewards5 ./ (V * 430), fillcolor = :red, linecolor = :red, legend = :outertopright, xlabel = "Episodes", ylabel = ylab, title = raw"$\mathrm{X_{0} =} {" * string(V * 430) * raw"}$", titlefontsize = 6, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5}$", legendfontsize = 4, fg_legend = :transparent, fontfamily="Computer Modern")
            plot!(rewards10 ./ (V * 430), fillcolor = :blue, linecolor = :blue, legend = :outertopright, xlabel = "Episodes", ylabel = ylab, title = raw"$\mathrm{X_{0} =} {" * string(V * 430) * raw"}$", titlefontsize = 6, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10}$", legendfontsize = 4, fg_legend = :transparent)
            hline!([10000], linecolor = :black, label = raw"$\mathrm{m_{0}}$", linestyle = :dash)

        else

            p = plot(rewards5 ./ (V * 430), fillcolor = :red, linecolor = :red, legend = :outertopright, xlabel = "Episodes", ylabel = ylab, title = raw"$\mathrm{X_{0} =} {" * string(V * 430) * raw"}$", titlefontsize = 6, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5}$", legendfontsize = 4, fg_legend = :transparent, fontfamily="Computer Modern")
            plot!(rewards10 ./ (V * 430), fillcolor = :blue, linecolor = :blue, legend = :outertopright, xlabel = "Episodes", ylabel = ylab, title = raw"$\mathrm{X_{0} =} {" * string(V * 430) * raw"}$", titlefontsize = 6, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10}$", legendfontsize = 4, fg_legend = :transparent)
            hline!([10000], linecolor = :black, label = raw"$\mathrm{m_{0}}$", linestyle = :dash)

        end
        
        push!(reward_plots, p)
    end

    reward_plot = plot(reward_plots..., layout = grid(3,1), guidefontsize = 5, tickfontsize = 5, fontfamily="Computer Modern")
    savefig(reward_plot, path_to_files * "/Images/RL/RewardConvergence430.pdf")

    # plot the convergence in the number of states and trades
    num_states = [5,10]
    Vs = [200, 100, 50]
    num_states_dict = Dict()
    num_trades_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            num_states = Vector{Float64}()
            num_trades = Vector{Float64}()
            for i in 1:n
                push!(num_states, length(l[i]["Q"]))
                push!(num_trades, l[i]["NumberTrades"])
            end
            push!(num_states_dict, string(num_state) * "_" * string(V) => num_states)
            push!(num_trades_dict, string(num_state) * "_" * string(V) => num_trades)
        end
    end
    num_states_plots = plot(1:length(num_states_dict["5_200"]), num_states_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent, xlabel = "Episodes", ylabel = raw"# States ($\mathrm{n_{T},n_{I},n_{S},n_{V} = 5}$)", legendfontsize = 8, right_margin = 15mm, fontfamily="Computer Modern")
    plot!(1:length(num_states_dict["5_100"]), num_states_dict["5_100"], fillcolor = :red, linecolor = :red, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_states_dict["5_50"]), num_states_dict["5_50"], fillcolor = :green, linecolor = :green, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent)
    subplot = twinx()
    plot!(subplot, 1:length(num_states_dict["10_200"]), num_states_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", ylabel = raw"# States ($\mathrm{n_{T},n_{I},n_{S},n_{V} = 10}$)", legend = :bottomright, fg_legend = :transparent, legendfontsize = 8)
    plot!(subplot, 1:length(num_states_dict["10_100"]), num_states_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :bottomright, fg_legend = :transparent)
    plot!(subplot, 1:length(num_states_dict["10_50"]), num_states_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :bottomright, fg_legend = :transparent)
    savefig(num_states_plots, path_to_files * "/Images/RL/NumberStatesConvergence430.pdf")

    num_trades_plots = plot(1:length(num_trades_dict["10_200"]), num_trades_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", legend = :bottomright, fg_legend = :transparent, legendfontsize = 8, xlabel = "Episodes", ylabel = "# Trades", fontfamily="Computer Modern")
    plot!(1:length(num_trades_dict["10_100"]), num_trades_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :bottomright, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["10_50"]), num_trades_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :bottomright, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_200"]), num_trades_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_100"]), num_trades_dict["5_100"], fillcolor = :red, linecolor = :red, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent)
    plot!(1:length(num_trades_dict["5_50"]), num_trades_dict["5_50"], fillcolor = :green, linecolor = :green, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :bottomleft, fg_legend = :transparent)
    savefig(num_trades_plots, path_to_files * "/Images/RL/NumberTradesConvergence430.pdf")

    # convergence of policy (difference between best action in each state in consecutive iterations)
    num_states = [5,10]
    Vs = [200, 100, 50]
    policy_diffs_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            p_diffs = Vector{Float64}()
            for i in 2:n
                prev_q = l[i-1]["Q"]
                prev_q_state_values = getindex.(Ref(l[i-1]["Q"]), keys(prev_q))
                curr_q_state_values = getindex.(Ref(l[i]["Q"]), keys(prev_q))
                prev_policy = argmax.(prev_q_state_values)
                current_policy = argmax.(curr_q_state_values)
                p_diff = sum(prev_policy .!= current_policy) / length(prev_policy) 
                push!(p_diffs, p_diff)
            end
            push!(policy_diffs_dict, string(num_state) * "_" * string(V) => p_diffs)
        end
    end

    policy_diffs_plots = plot(1:length(policy_diffs_dict["5_200"]), policy_diffs_dict["5_200"], fillcolor = :blue, linecolor = :blue, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", fg_legend = :transparent, legendfontsize = 8, xlabel = "Episodes", ylabel = "One Step Policy Difference", fontfamily="Computer Modern")
    plot!(1:length(policy_diffs_dict["5_100"]), policy_diffs_dict["5_100"], fillcolor = :red, linecolor = :red, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(100 * 430) * raw"}$")
    plot!(1:length(policy_diffs_dict["5_50"]), policy_diffs_dict["5_50"], fillcolor = :green, linecolor = :green, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(50 * 430) * raw"}$")
    plot!(1:length(policy_diffs_dict["10_200"]), policy_diffs_dict["10_200"], fillcolor = :magenta, linecolor = :magenta, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(200 * 430) * raw"}$")
    plot!(1:length(policy_diffs_dict["10_100"]), policy_diffs_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(100 * 430) * raw"}$")
    plot!(1:length(policy_diffs_dict["10_50"]), policy_diffs_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(50 * 430) * raw"}$")
    savefig(policy_diffs_plots, path_to_files * "/Images/RL/PolicyConvergence430.pdf")

    # convergence of Q (difference between best action in each state in consecutive iterations)
    num_states = [5,10]
    Vs = [200, 100, 50]
    q_diffs_dict = Dict()
    for num_state in num_states
        for V in Vs
            l = data["Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state)]
            n = length(l)
            # convergence of q
            q_diffs = Vector{Float64}()
            for i in 2:n
                prev_q = l[i-1]["Q"]
                prev_q_state_values = getindex.(Ref(l[i-1]["Q"]), keys(prev_q))
                curr_q_state_values = getindex.(Ref(l[i]["Q"]), keys(prev_q))
                q_diff = [sum(abs.(s_diff)) / length(prev_q_state_values[1]) for s_diff in (curr_q_state_values .- prev_q_state_values)]
                push!(q_diffs, sum(q_diff) / length(prev_q))
            end
            push!(q_diffs_dict, string(num_state) * "_" * string(V) => q_diffs)
        end
    end
    
    q_diffs_plots = plot(1:length(q_diffs_dict["5_200"]), q_diffs_dict["5_200"], ylims = (0, 150000), fillcolor = :blue, linecolor = :blue, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", legend = :topleft, fg_legend = :transparent, xlabel = "Episodes", ylabel = raw"One Step Q-matrix Difference ($\mathrm{n_{T},n_{I},n_{S},n_{V} = 5}$)", legendfontsize = 6, guidefontsize = 8, tickfontsize = 7, right_margin = 25mm, fontfamily="Computer Modern")
    plot!(1:length(q_diffs_dict["5_100"]), q_diffs_dict["5_100"], fillcolor = :red, linecolor = :red, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :topleft, fg_legend = :transparent)
    plot!(1:length(q_diffs_dict["5_50"]), q_diffs_dict["5_50"], fillcolor = :green, linecolor = :green, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 5, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :topleft, fg_legend = :transparent)
    subplot = twinx()
    plot!(subplot, 1:length(q_diffs_dict["10_200"]), q_diffs_dict["10_200"], ylims = (0, 13000), fillcolor = :magenta, linecolor = :magenta, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(200 * 430) * raw"}$", legend = :topright, fg_legend = :transparent, ylabel = raw"One Step Q-matrix Difference ($\mathrm{n_{T},n_{I},n_{S},n_{V} = 10}$)", legendfontsize = 6, guidefontsize = 8, tickfontsize = 7)
    plot!(subplot, 1:length(q_diffs_dict["10_100"]), q_diffs_dict["10_100"], fillcolor = :orange, linecolor = :orange, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(100 * 430) * raw"}$", legend = :topright, fg_legend = :transparent)
    plot!(subplot, 1:length(q_diffs_dict["10_50"]), q_diffs_dict["10_50"], fillcolor = :purple, linecolor = :purple, label = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V} = 10, \;\; X_{0} =} {" * string(50 * 430) * raw"}$", legend = :topright, fg_legend = :transparent)
    savefig(q_diffs_plots, path_to_files * "/Images/RL/QConvergence430.pdf")

end
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# PlotRLConvergenceResults(actions)
#---------------------------------------------------------------------------------------------------

#----- State-action convergence -----# 
function StateActionConvergence(l::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, V::Int64, actionsMap::Dict)

    n = length(l)

    # get max number of states (last iteration states)
    max_states = collect(keys(l[1000]["Q"]))

    # for each key get the policy over the iterations (if state does not exist then -1)
    actions_dict = Dict()
    for state in max_states
        actions = Vector{Float64}()
        for i in 1:n
            if state in collect(keys(l[i]["Q"]))
                push!(actions, actionsMap[argmax(l[i]["Q"][state])])
            else
                push!(actions, -1)
            end
        end
        push!(actions_dict, state => actions)
    end

    p = plot(actions_dict[max_states[1]], legend = false, xlabel = "Episodes", ylabel = "Actions", fontfamily = "Computer Modern", title = raw"$\mathrm{n_{T},n_{I},n_{S},n_{V}} = {" * string(numT) * raw"} \;\; (\mathrm{X_{0}} = {" * string(430 * V) * raw"})$", titlefontsize = 11, yticks = [-1;collect(range(0,2,9))])
    for i in 2:length(max_states)
        plot!(actions_dict[max_states[i]])
    end
    savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V" * string(V) * "_S" * string(numT) * "_430/StateActionConvergence_V" * string(V) * "_S" * string(numT) * "_430.pdf")

end
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = I = B = W = 5                    # number of time, inventory, spread, volume states 
# V = 50
# @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(numT) * "_430.jld")["rl_results"]
# n = length(l)
# StateActionConvergence(l, numT, I, B, W, A, V, actions)
#---------------------------------------------------------------------------------------------------

#----- Given a Q-matrix get the greedy policy -----# 
function GetPolicy(Q::Dict)
    P = Dict{Vector{Int64}, Int64}()
    for state in collect(keys(Q))
        push!(P, state => argmax(Q[state]))
    end
    return P 
end
#---------------------------------------------------------------------------------------------------

#----- Visualize the a single agents policy -----# 
function PolicyVisualization(Q::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, V::Int64, actionsMap::Dict)

    P = GetPolicy(Q)
    plots = []
    inc = 1
    for i in I:-1:1 # i in I:-1:1 # want volume to increase upwards in plot
        for t in numT:-1:1 # t in numT:-1:1 # want time remaining to decrease left to right
            # create a matrix that will store values for spread and volume states
            M = fill(0.0,B,W)
            s_counter = 1
            for s in 1:1:B # s in 1:1:B
                v_counter = 1
                for v in 1:1:W # v in 1:1:W
                    # for each of these states get the action associted with it, if it does not exist then -1
                    key = [t, i, s, v]
                    M[s_counter,v_counter] = -1
                    if key in collect(keys(P))
                        M[s_counter,v_counter] = actionsMap[P[key]]
                    end
                    v_counter += 1
                end
                s_counter += 1
            end
            # for a given t and i plot the actions taken over the spread and volume states
            xlabel = ""
            ylabel = ""
            if t == numT && i == I # t == 5 && i == 5 specify the x and y labels for each individual heatmap
                xlabel = "Volume"
                ylabel = "Spread"
            end
            h = heatmap(1:B, 1:W, M, xlabel = xlabel, ylabel = ylabel, c = cgrad(:seismic, [0, 0.50, 0.78, 1]), clim = (-1, actionsMap[A]), guidefontsize = 4, tick_direction = :out, legend = false, tickfontsize = 4, margin = -1mm)
            push!(plots, h)
        end
    end
    l = @layout[a{0.05w} grid(5,5); b{0.001h}]
    colorbar = heatmap([-1;getindex.(Ref(actionsMap), 1:A)].*ones(A+1,1), title = "Actions", titlefontsize = 7, ylabel = "Inventory", ymirror = true, guidefontsize = 10, tickfontsize = 5, c = cgrad(:seismic, [0, 0.50, 0.78, 1]), legend=:none, xticks=:none, yticks=(1:1:(A+1), string.([-1;getindex.(Ref(actionsMap), 1:A)])), y_foreground_color_axis=:white, y_foreground_color_border=:white)
    empty = plot(title = "Time", titlefontsize = 10, legend=false,grid=false, foreground_color_axis=:white, foreground_color_border=:white, ticks = :none)
    p = plot(colorbar, plots..., empty, layout = l, fontfamily = "Computer Modern")
    # savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V" * string(V) * "_S" * string(numT) * "_430/PolicyPlot_V" * string(V) * "_S" * string(numT) * "_430.pdf")
    savefig(p, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/PolicyPlot_V" * string(V) * "_S" * string(numT) * "_430.pdf")

end
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# V = 100
# numT = I = B = W = 5                    # number of time, inventory, spread, volume states
# # @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(numT) * "_430.jld")["rl_results"]
# @time l = load(path_to_files * "Data/RL/Training/ResultsTest_Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25.jld")["rl_results"]
# n = length(l)
# PolicyVisualization(l[200]["Q"], numT, I, B, W, A, V, actions)
# #---------------------------------------------------------------------------------------------------

#----- Agents Actions per State Value (averaged ove other states) -----# 
function AverageActionsPerStateValue(Q::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, actionsMap::Dict)

    states = collect(keys(Q))

    # get average actions per time value
    avg_action_time = Vector{Float64}() # time remaining increases from start to finish
    for t in 1:numT
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[1] == t, states)]))
        push!(avg_action_time, mean(getindex.(Ref(actionsMap), action_ids)))
    end

    # get average actions per inventory value
    avg_action_inventory = Vector{Float64}() # time remaining increases from start to finish
    for i in 1:I
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[2] == i, states)]))
        push!(avg_action_inventory, mean(getindex.(Ref(actionsMap), action_ids)))
    end

    # get average actions per spread value
    avg_action_spread = Vector{Float64}() # time remaining increases from start to finish
    for s in 1:B
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[3] == s, states)]))
        push!(avg_action_spread, mean(getindex.(Ref(actionsMap), action_ids)))
    end

    # get average actions per volume value
    avg_action_volume = Vector{Float64}() # time remaining increases from start to finish
    for v in 1:W
        action_ids = argmax.(getindex.(Ref(Q), states[findall(x -> x[4] == v, states)]))
        push!(avg_action_volume, mean(getindex.(Ref(actionsMap), action_ids)))
    end

    # plot the effects
    p = plot(reverse(avg_action_time), label = "time", legend = :outertopright, fg_legend = :transparent, ylabel = "Average Action", xlabel = "State Value")
    plot!(avg_action_inventory, label = "inventory")
    plot!(avg_action_spread, label = "spread")
    plot!(avg_action_volume, label = "volume")

    savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V50_S5_430/AverageActionEffects_V50_S5_430.pdf")

end
# @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V50_S5_430.jld")["rl_results"]
# n = length(l)
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = I = B = W = 5                    # number of time, inventory, spread, volume states
# AverageActionsPerStateValue(l[1000]["Q"], numT, I, B, W, A, actions)
#---------------------------------------------------------------------------------------------------

#----- Percent Correct Actions -----# 
function PlotAverageActionsOverTime(l::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, V::Int64, actionsMap::Dict)

    n = length(l)
    
    # get the average actions
    avg_actions_highvol = Vector{Float64}()
    avg_actions_lowvol = Vector{Float64}()

    avg_actions_highspr = Vector{Float64}()
    avg_actions_lowspr = Vector{Float64}()

    avg_actions_hightime = Vector{Float64}()
    avg_actions_lowtime = Vector{Float64}()

    avg_actions_highinv = Vector{Float64}()
    avg_actions_lowinv = Vector{Float64}()

    avg_actions_highvol_lowspr = Vector{Float64}()
    avg_actions_lowvol_highspr = Vector{Float64}()

    avg_actions_hightime_highinv = Vector{Float64}()
    avg_actions_lowtime_lowinv = Vector{Float64}()

    for i in 1:n
        states = collect(keys(l[i]["Q"]))
        # get the required states
        if numT == 10

            highvol = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] >= 7 && x[1] > 0, states)])
            lowvol = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] <= 4 && x[1] > 0, states)])

            highspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[3] >= 6 && x[1] > 0, states)])
            lowspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[3] <= 4 && x[1] > 0, states)])

            hightime = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] >= 7 && x[1] > 0, states)])
            lowtime = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] <= 4 && x[1] > 0, states)])

            highinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[2] >= 7 && x[1] > 0, states)])
            lowinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[2] <= 4 && x[1] > 0, states)])

            highvol_lowspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] >= 7 && x[3] <= 4 && x[1] > 0, states)]) 
            lowvol_highspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] <= 4 && x[3] >= 6 && x[1] > 0, states)]) 

            hightime_highinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] >= 7 && x[2] >= 7 && x[1] > 0, states)]) 
            lowtime_lowinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] <= 4 && x[2] <= 4 && x[1] > 0, states)]) 

            # remove the states just visited [0,0,...,0], and get the actions
            avg_action_highvol = mean(getindex.(Ref(actionsMap) ,argmax.(highvol[findall(x -> x != zeros(9), highvol)])))
            avg_action_lowvol = mean(getindex.(Ref(actionsMap) ,argmax.(lowvol[findall(x -> x != zeros(9), lowvol)])))

            avg_action_highspr = mean(getindex.(Ref(actionsMap) ,argmax.(highspr[findall(x -> x != zeros(9), highspr)])))
            avg_action_lowspr = mean(getindex.(Ref(actionsMap) ,argmax.(lowspr[findall(x -> x != zeros(9), lowspr)])))

            avg_action_hightime = mean(getindex.(Ref(actionsMap) ,argmax.(hightime[findall(x -> x != zeros(9), hightime)])))
            avg_action_lowtime = mean(getindex.(Ref(actionsMap) ,argmax.(lowtime[findall(x -> x != zeros(9), lowtime)])))

            avg_action_highinv = mean(getindex.(Ref(actionsMap) ,argmax.(highinv[findall(x -> x != zeros(9), highinv)])))
            avg_action_lowinv = mean(getindex.(Ref(actionsMap) ,argmax.(lowinv[findall(x -> x != zeros(9), lowinv)])))

            avg_action_highvol_lowspr = mean(getindex.(Ref(actionsMap) ,argmax.(highvol_lowspr[findall(x -> x != zeros(9), highvol_lowspr)])))
            avg_action_lowvol_highspr = mean(getindex.(Ref(actionsMap) ,argmax.(lowvol_highspr[findall(x -> x != zeros(9), lowvol_highspr)])))

            avg_action_hightime_highinv = mean(getindex.(Ref(actionsMap) ,argmax.(hightime_highinv[findall(x -> x != zeros(9), hightime_highinv)])))
            avg_action_lowtime_lowinv = mean(getindex.(Ref(actionsMap) ,argmax.(lowtime_lowinv[findall(x -> x != zeros(9), lowtime_lowinv)])))

            # add to vector
            push!(avg_actions_highvol, avg_action_highvol)
            push!(avg_actions_lowvol, avg_action_lowvol)

            push!(avg_actions_highspr, avg_action_highspr)
            push!(avg_actions_lowspr, avg_action_lowspr)

            push!(avg_actions_hightime, avg_action_hightime)
            push!(avg_actions_lowtime, avg_action_lowtime)

            push!(avg_actions_highinv, avg_action_highinv)
            push!(avg_actions_lowinv, avg_action_lowinv)

            push!(avg_actions_highvol_lowspr, avg_action_highvol_lowspr)
            push!(avg_actions_lowvol_highspr, avg_action_lowvol_highspr)

            push!(avg_actions_hightime_highinv, avg_action_hightime_highinv)
            push!(avg_actions_lowtime_lowinv, avg_action_lowtime_lowinv)
        else

            highvol = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] >= 4 && x[1] > 0, states)])
            lowvol = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] <= 2 && x[1] > 0, states)])

            highspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[3] >= 3 && x[1] > 0, states)])
            lowspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[3] <= 2 && x[1] > 0, states)])

            hightime = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] >= 4 && x[1] > 0, states)])
            lowtime = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] <= 2 && x[1] > 0, states)])

            highinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[2] >= 4 && x[1] > 0, states)])
            lowinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[2] <= 2 && x[1] > 0, states)])

            highvol_lowspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] >= 4 && x[3] <= 2 && x[1] > 0, states)]) 
            lowvol_highspr = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[4] <= 2 && x[3] >= 3 && x[1] > 0, states)])

            hightime_highinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] >= 4 && x[2] >= 4 && x[1] > 0, states)]) 
            lowtime_lowinv = getindex.(Ref(l[i]["Q"]), states[findall(x -> x[1] <= 2 && x[2] <= 2 && x[1] > 0, states)]) 

            # remove the states just visited [0,0,...,0], and get the actions
            avg_action_highvol = mean(getindex.(Ref(actionsMap) ,argmax.(highvol[findall(x -> x != zeros(9), highvol)])))
            avg_action_lowvol = mean(getindex.(Ref(actionsMap) ,argmax.(lowvol[findall(x -> x != zeros(9), lowvol)])))

            avg_action_highspr = mean(getindex.(Ref(actionsMap) ,argmax.(highspr[findall(x -> x != zeros(9), highspr)])))
            avg_action_lowspr = mean(getindex.(Ref(actionsMap) ,argmax.(lowspr[findall(x -> x != zeros(9), lowspr)])))

            avg_action_hightime = mean(getindex.(Ref(actionsMap) ,argmax.(hightime[findall(x -> x != zeros(9), hightime)])))
            avg_action_lowtime = mean(getindex.(Ref(actionsMap) ,argmax.(lowtime[findall(x -> x != zeros(9), lowtime)])))

            avg_action_highinv = mean(getindex.(Ref(actionsMap) ,argmax.(highinv[findall(x -> x != zeros(9), highinv)])))
            avg_action_lowinv = mean(getindex.(Ref(actionsMap) ,argmax.(lowinv[findall(x -> x != zeros(9), lowinv)])))

            avg_action_highvol_lowspr = mean(getindex.(Ref(actionsMap) ,argmax.(highvol_lowspr[findall(x -> x != zeros(9), highvol_lowspr)])))
            avg_action_lowvol_highspr = mean(getindex.(Ref(actionsMap) ,argmax.(lowvol_highspr[findall(x -> x != zeros(9), lowvol_highspr)])))

            avg_action_hightime_highinv = mean(getindex.(Ref(actionsMap) ,argmax.(hightime_highinv[findall(x -> x != zeros(9), hightime_highinv)])))
            avg_action_lowtime_lowinv = mean(getindex.(Ref(actionsMap) ,argmax.(lowtime_lowinv[findall(x -> x != zeros(9), lowtime_lowinv)])))

            # add to vector
            push!(avg_actions_highvol, avg_action_highvol)
            push!(avg_actions_lowvol, avg_action_lowvol)

            push!(avg_actions_highspr, avg_action_highspr)
            push!(avg_actions_lowspr, avg_action_lowspr)

            push!(avg_actions_hightime, avg_action_hightime)
            push!(avg_actions_lowtime, avg_action_lowtime)

            push!(avg_actions_highinv, avg_action_highinv)
            push!(avg_actions_lowinv, avg_action_lowinv)

            push!(avg_actions_highvol_lowspr, avg_action_highvol_lowspr)
            push!(avg_actions_lowvol_highspr, avg_action_lowvol_highspr)

            push!(avg_actions_hightime_highinv, avg_action_hightime_highinv)
            push!(avg_actions_lowtime_lowinv, avg_action_lowtime_lowinv)

        end
        
    end

    numT == 5 ? legend_symbol = Symbol("topright") : legend_symbol = Symbol("bottomleft")

    # plot the volume  
    v = plot(avg_actions_highvol, label = "High Volume", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_lowvol, label = "Low Volume", color = :red, fg_legend = :transparent)
    savefig(v, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionVolume_V" * string(V) * "_S" * string(numT) * "_430.pdf")

    # plot the spread
    s = plot(avg_actions_lowspr, label = "Narrow Spread", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_highspr, label = "Wide Spread", color = :red, fg_legend = :transparent)
    savefig(s, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionSpread_V" * string(V) * "_S" * string(numT) * "_430.pdf")

    # plot the time
    t = plot(avg_actions_hightime, label = "More remaining time", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_lowtime, label = "Less remaining time", color = :red, fg_legend = :transparent)
    savefig(t, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionTime_V" * string(V) * "_S" * string(numT) * "_430.pdf")

    # plot the inventory
    i = plot(avg_actions_highinv, label = "More remaining inventory", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_lowinv, label = "Less remaining inventory", color = :red, fg_legend = :transparent)
    savefig(i, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionInventory_V" * string(V) * "_S" * string(numT) * "_430.pdf")

    # plot the spread vs volume interaction 
    sv = plot(avg_actions_highvol_lowspr, label = "High Volume, Narrow Spread", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_lowvol_highspr, label = "Low Volume, Wide Spread", color = :red, fg_legend = :transparent)
    savefig(sv, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionSpreadVolume_V" * string(V) * "_S" * string(numT) * "_430.pdf")

    # plot the time and inventory interactions
    ti = plot(avg_actions_hightime_highinv, label = "More remaining time, more remaining inventory", color = :blue, legend = legend_symbol, fg_legend = :transparent, xlabel = "Episodes", ylabel = "Average Action", title = raw"$\mathrm{X_{0} =} {" * string(430 * V) * raw"} \;\; \mathrm{(n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"}$)", titlefontsize = 11, fontfamily = "Computer Modern")
    plot!(avg_actions_lowtime_lowinv, label = "Less remaining time, less remaining inventory", color = :red, fg_legend = :transparent)
    savefig(ti, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/AverageActionTimeInventory_V" * string(V) * "_S" * string(numT) * "_430.pdf")


end
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = I = B = W = 5                    # number of time, inventory, spread, volume states 
# V = 100
# @time l = load(path_to_files * "Data/RL/Training/ResultsTest_Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25.jld")["rl_results"]
# n = length(l)
# PlotAverageActionsOverTime(l, numT, I, B, W, A, V, actions)
#---------------------------------------------------------------------------------------------------

#----- Percent Correct Actions -----# 
function PlotVolumeTragectory(l::Dict, numT::Int64, I::Int64, B::Int64, W::Int64, A::Int64, V::Int64, actionsMap::Dict)
    
    # plot the initial actions and the final actions to see if there are differences in actions selected
    actions1 = Vector{Float64}()
    for action in l[1]["Actions"]
        push!(actions1, action)
    end
    pi1 = plot(1:l[1]["NumberActions"], getindex.(Ref(actionsMap), actions1) .* V, size = (800, 400), seriestype = :line, fillcolor = :blue, linecolor = :blue, legend = false, xlabel = "Action Number", ylabel = "Action volume", title = raw"Episode 1 ($\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * V) * raw"}$)", titlefontsize = 9, guidefontsize = 8, tickfontsize = 8, left_margin = 5mm, bottom_margin = 5mm)
    
    actionsN = Vector{Float64}()
    n = 200
    for action in l[n]["Actions"]
        push!(actionsN, action)
    end
    piN = plot(1:l[n]["NumberActions"], getindex.(Ref(actionsMap), actionsN) .* V, size = (800, 400), seriestype = :line, fillcolor = :blue, linecolor = :blue, legend = false, xlabel = "Action Number", ylabel = "Action volume", title = raw"Episode " * string(n) * raw" ($\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(numT) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * V) * raw"}$)", titlefontsize = 9, guidefontsize = 8, tickfontsize = 8, left_margin = 5mm, bottom_margin = 5mm)
    l = @layout([a; b])
    p = plot(pi1, piN, layout = l, fontfamily = "Computer Modern")
    # savefig(p, path_to_files * "/Images/RL/alpha0.1_iteration1000_V" * string(V) * "_S" * string(numT) * "_430/ActionVolumeIteration_" * string(n) * "_V" * string(V) * "_S" * string(numT) * "_430.pdf")
    savefig(p, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/ActionVolumeIteration_" * string(n) * "_V" * string(V) * "_S" * string(numT) * "_430.pdf")
end
# A = 9                          # number of action states (if odd TWAP price will be an option else it will be either higher or lower)
# maxVolunmeIncrease = 2.0       # maximum increase in the number of TWAP shares (fix at 2 to make sure there are equal choices to increase and decrease TWAP volume)
# actions = GenerateActions(A, maxVolunmeIncrease)
# numT = I = B = W = 5                    # number of time, inventory, spread, volume states 
# V = 100
# # @time l = load(path_to_files * "Data/RL/Training/Results_alpha0.1_iterations1000_V" * string(V) * "_S" * string(numT) * "_430.jld")["rl_results"]
# @time l = load(path_to_files * "Data/RL/Training/ResultsTest_Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25.jld")["rl_results"]
# n = length(l)
# PlotVolumeTragectory(l, numT, I, B, W, A, V, actions)
# ---------------------------------------------------------------------------------------------------

# @time l = load(path_to_files * "Data/RL/Training/ResultsTest_Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25.jld")["rl_results"]
# rewards = [l[i]["TotalReward"] for i in 1:200]
# # rewards = rewards[findall(x -> abs(x) < 5000000, rewards)]
# p = plot(rewards, fillcolor = :red, linecolor = :red, legend = :outertopright, xlabel = "Episodes", legendfontsize = 4, fg_legend = :transparent, fontfamily="Computer Modern")
# # hline!(p, [(-10000 + 10100) * 43000], color = :black)
# savefig(p, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/Reward.pdf")

# num_trades = [l[i]["NumberTrades"] for i in 1:200]
# p = plot(num_trades, fillcolor = :red, linecolor = :red, legend = :outertopright, xlabel = "Episodes", legendfontsize = 4, fg_legend = :transparent, fontfamily="Computer Modern")
# # hline!(p, [(-10000 + 10100) * 43000], color = :black)
# savefig(p, path_to_files * "/Images/RL/Slippage_Sell_ExpTIDivXPenalty_RLOutVWAP_E200_alpha0.5_lambda0.001_gamma0.25/NumTrades.pdf")

#----------------------------------------------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------- Stylised Fact Visualizations ------------------------------------------------------#

#----- Generate emperical log-returns and emperical moments -----#
function GenerateEmpericalReturnsAndMoments(startTime::DateTime, endTime::DateTime)
    println("Generating returns and moments for: " * Dates.format(startTime, "yyyy-mm-ddTHH:MM:SS") * " to " * Dates.format(endTime, "yyyy-mm-ddTHH:MM:SS"))
    empericalData = CSV.File(path_to_files * string("/Data/JSE/L1LOB.csv"), missingstring = "missing", types = Dict(:DateTime => DateTime, :Type => Symbol)) |> DataFrame
    filter!(x -> startTime <= x.DateTime && x.DateTime < endTime, empericalData)
    filter!(x -> !ismissing(x.MidPrice), empericalData); filter!(x -> !ismissing(x.MicroPrice), empericalData)
    midPriceLogReturns = diff(log.(empericalData.MidPrice))
    microPriceLogReturns = diff(log.(empericalData.MicroPrice))
    empericalLogReturns = DataFrame(MidPriceLogReturns = midPriceLogReturns, MicroPriceLogReturns = microPriceLogReturns)
    empericalMidPriceMoments = Moments(midPriceLogReturns, midPriceLogReturns)
    empericalMicroPriceMoments = Moments(microPriceLogReturns, microPriceLogReturns)
    empericalMoments = Dict("empericalMidPriceMoments" => empericalMidPriceMoments, "empericalMicroPriceMoments" => empericalMicroPriceMoments)
    return empericalLogReturns, empericalMoments
end
#---------------------------------------------------------------------------------------------------

#----- Return the mid and micro prices -----#
function GetSimulatedMidMicroPrices(cleanDirPath::String, rawDataFilePath::String, iteration::Int64)
    println("-------------------------------- Generating Mid and Micro Prices --------------------------------")

    # copy raw data file to cleaning directory
    run(`cp $rawDataFilePath $cleanDirPath`)

    # clean the data
    CleanData("RawRLIteration" * string(iteration), initialization = false)

    # get the l1lob data and then move L1LOB data file back to original directory
    l1lob_file_path = cleanDirPath * "L1LOBRLIteration" * string(iteration) * ".csv"
    l1lob_data = CSV.File(l1lob_file_path, drop = [:Spread, :Price], missingstring = "missing", types = Dict(:DateTime => DateTime, :Initialization => Symbol, :Type => Symbol)) |> x -> filter(y -> y.Initialization != :INITIAL, x) |> DataFrame
    raw_data_dir = join(split(rawDataFilePath,"/")[1:(end-1)], "/")
    run(`mv $l1lob_file_path $raw_data_dir`)

    # move depth profile data to original directory
    depth_file_path = cleanDirPath * "/DepthProfileDataRLIteration" * string(iteration) *".csv"
    run(`mv $depth_file_path $raw_data_dir`)

    # delete depth profile, TAQ data and Raw data file from cleaning directory
    taq_file_path =  cleanDirPath * "/TAQRLIteration" * string(iteration) *".csv"
    raw_file_path = cleanDirPath * "/RawRLIteration" * string(iteration) *".csv"
    run(`rm $taq_file_path`)
    run(`rm $raw_file_path`)

    # from the L1LOB file get the mid and micro prices
    return collect(skipmissing(l1lob_data[:,:MidPrice])), collect(skipmissing(l1lob_data[:,:MicroPrice]))

end
#---------------------------------------------------------------------------------------------------

#----- Plot the stylised facts over the training period -----#
function PlotStylisedFactDynamics(empiricalLogReturns::DataFrame, moments::Vector{String}, abmMomentsdf::DataFrame)
    
    # for each volume and state space combination
    Vs = [50, 100, 200]
    num_states = [5, 10]

    # storage for the iterations
    iterations = round.(Int64, [1;collect(range(0,1000,11)[2:end])])

    stylised_fact_data = Dict()
    for V in Vs
        for num_state in num_states
            path = path_to_files * "Data/CoinTossX/alpha0.1_iterations1000_V" * string(V) * "_S" * string(num_state) * "_430"

            # create dict to store moment data
            moment_dict = Dict{String,Vector{Float64}}()
            for moment in moments
                push!(moment_dict, moment => [])
            end

            # for each of the iterations
            for iteration in iterations
                file_path = path * "/RawRLIteration" * string(iteration) * ".csv"
                mid_prices, micro_prices = GetSimulatedMidMicroPrices(path_to_files * "Data/CoinTossX/", file_path, iteration)
            
                # compute the moments and store the results
                simulated_micro_price_moments = Moments(diff(log.(micro_prices)), empiricalLogReturns.MicroPriceLogReturns)
                push!(moment_dict[moments[1]], simulated_micro_price_moments.μ); push!(moment_dict[moments[2]], simulated_micro_price_moments.σ); 
                push!(moment_dict[moments[3]], simulated_micro_price_moments.ks); push!(moment_dict[moments[4]], simulated_micro_price_moments.hurst); push!(moment_dict[moments[5]], simulated_micro_price_moments.gph); 
                push!(moment_dict[moments[6]], simulated_micro_price_moments.adf); push!(moment_dict[moments[7]], simulated_micro_price_moments.garch); push!(moment_dict[moments[8]], simulated_micro_price_moments.hill);     
            end

            # add to the agents dict
            push!(stylised_fact_data, string(V) * "_" * string(num_state) => moment_dict)

        end
    end

    # for each moment plot the stylised facts over time
    colors = [:green, :magenta, :purple, :orange, :lightgreen, :grey9]
    for i in 1:nrow(abmMomentsdf)
        color_count = 1
        p = plot(iterations, round.(stylised_fact_data["50_5"][moments[i]], digits = 3), size = (800, 400), ylabel = moments[i], color = colors[color_count], label = raw"RL ($\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(5) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * 50) * raw"}$)", xlabel = "Iterations", marker = :circle, markerstrokewidth = 0, markersize = 3, legend = :outertopright, legendfontsize = 7, fg_legend = :transparent, titlefontsize = 11, left_margin = 5mm, bottom_margin = 5mm, fontfamily = "Computer Modern")
        for V in Vs
            for num_state in num_states
                if V == 50 && num_state == 5
                    color_count += 1
                    continue
                else
                    plot!(iterations, round.(stylised_fact_data[string(V) * "_" * string(num_state)][moments[i]], digits = 3), size = (800, 400), ylabel = moments[i], color = colors[color_count], label = raw"RL ($\mathrm{n_{T},n_{I},n_{S},n_{V} =} {" * string(num_state) * raw"} \;\; \mathrm{X_{0} =} {" * string(430 * V) * raw"}$)", xlabel = "Iterations", marker = :circle, markerstrokewidth = 0, markersize = 3, legend = :outertopright, legendfontsize = 7, fg_legend = :transparent, titlefontsize = 11, left_margin = 5mm, bottom_margin = 5mm)
                end
                color_count += 1
            end
        end
        hline!([round(abmMomentsdf[i,"Simulated"], digits = 3)], label = "ABM", color = :blue)
        hline!([round(abmMomentsdf[i,"SimulatedLower"], digits = 3)], color = :blue, linestyle = :dash, label = "")
        hline!([round(abmMomentsdf[i,"SimulatedUpper"], digits = 3)], color = :blue, linestyle = :dash, label = "")
        hline!([round(abmMomentsdf[i,"Empirical"], digits = 3)], label = "JSE", color = :red)
        hline!([round(abmMomentsdf[i,"EmpiricalLower"], digits = 3)], color = :red, linestyle = :dash, label = "")
        hline!([round(abmMomentsdf[i,"EmpiricalUpper"], digits = 3)], color = :red, linestyle = :dash, label = "")
        savefig(p, path_to_files * "/Images/RL/MomentDynamics_" * moments[i] * "_430.pdf")
    end 

    # change directory back to current directory
    cd(path_to_files * "/Scripts")

end

# # make sure these are the same for the stylized facts and Calibration
# date = DateTime("2019-07-08")
# startTime = date + Hour(9) + Minute(1)
# endTime = date + Hour(16) + Minute(50) # Hour(17) ###### Change to 16:50
# empericalLogReturns, empericalMoments = GenerateEmpericalReturnsAndMoments(startTime, endTime)

# # get the moment names and the ABM moments
# moments = ["Mean", "Std", "KS", "Hurst", "GPH", "ADF", "GARCH", "Hill"]
# abm_moments_df = CSV.File(path_to_files * "/Data/Calibration/moments.csv") |> DataFrame

# PlotStylisedFactDynamics(empericalLogReturns, moments, abm_moments_df)
#---------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------------------------------------#