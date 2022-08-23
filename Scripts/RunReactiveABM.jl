#=
RunReactiveABM:
- Julia Version: 1.7.1 
- Authors: Matthew Dicks, Tim Gebbie
- Function: Perform a single run of the ReactiveABM.jl for a single set of parameters (ASSUMES NO RL AGENTS)
- Structure: 
    1. Start the jvm and log into CoinTossX
    2. Initialise parameters
    3. Run the simulation
    4. Log out of CoinTossX
- Examples:
    StartJVM()
    gateway = Login(1,1)
    seed = 1
    parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)
    rlParameters = RLParameters(0, Dict(), Millisecond(0), Millisecond(0), 0, 0, 0, 0, 0, 0, 0, Dict(), DataFrame(), DataFrame(), "", 0.0, 0.0, 0)
    print_and_plot = true                    
    write = true 
    write_volume_spread = false
    rlTraders = false                        # should always be false in this file
    rlTraining = false                       # should always be false in this file
    @time simulate(parameters, rlParameters, gateway, rlTraders, rlTraining, print_and_plot, write_messages, write_volume_spread, seed = seed, iteration = 0)
    Logout(gateway)
- Prerequisites:
    1. CoinTossX is running
=#
ENV["JULIA_COPY_STACKS"]=1
path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/ReactiveABM.jl");  include(path_to_files * "Scripts/CoinTossXUtilities.jl") # also includes CoinTossXUtilities.jl

# set the parameters
Nᴸₜ = 8             # [3,6,9,12]
Nᴸᵥ = 6
Nᴴ = 30             # fixed at 30
δ = 0.125           # 0.01, 0.07, 0.14, 0.2
κ = 3.389           # 2, 3, 4, 5
ν = 7.221               # 2, 4, 6, 8
m₀ = 10000          # fixed at 10000
σᵥ = 0.041         # 0.0025, 0.01, 0.0175, 0.025
λmin = 0.0005       # fixed at 0.0005
λmax = 0.05         # fixed at 0.05
γ = Millisecond(1000) # fixed at 1000
T = Millisecond(25000) # fixed at 25000 
seed = 1

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T)

# set rl parameters to nothing 
rlParameters = RLParameters(0, Dict(), Millisecond(0), Millisecond(0), 0, 0, 0, 0, 0, 0, 0, Dict(), DataFrame(), DataFrame(), "", 0.0, 0.0, 0)

# set the parameters that dictate output
print_and_plot = true                    # Print out useful info about sim and plot simulation time series info
write_messages = true                    # Says whether or not the messages data must be written to a file
write_volume_spread = false
rlTraders = false                        # should always be false in this file
rlTraining = false                       # should always be false in this file

# run the simulation
StartJVM()
gateway = Login(1,1)
try 
    @time simulate(parameters, rlParameters, gateway, rlTraders, rlTraining, print_and_plot, write_messages, write_volume_spread, seed = seed, iteration = 0)
catch e
    @error "Something went wrong" exception=(e, catch_backtrace())
finally
    Logout(gateway)
end