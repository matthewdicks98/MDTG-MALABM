ENV["JULIA_COPY_STACKS"]=1
path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/ReactiveABM.jl");  include(path_to_files * "Scripts/CoinTossXUtilities.jl") # also includes CoinTossXUtilities.jl

# start CTX, start the JVM and 
# StartCoinTossX(build = false, deploy = true) # always redeploy to free up memory after last simulation
# sleep(5)

StartJVM()

gateway = Login(1,1)

# set the parameters
Nᴸₜ = 9             # [3,6,9,12]
Nᴸᵥ = 6
Nᴴ = 30             # fixed at 30
δ = 0.1            # 0.01, 0.07, 0.14, 0.2
κ = 2               # 2, 3, 4, 5
ν = 8               # 2, 4, 6, 8
m₀ = 10000          # fixed at 10000
σᵥ = 0.01          # 0.0025, 0.01, 0.0175, 0.025
λmin = 0.0005       # fixed at 0.0005
λmax = 0.05         # fixed at 0.05
γ = Millisecond(1000) # fixed at 1000
T = Millisecond(25000) # fixed at 25000 
seed = 1 # 125 has price decrease

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T, seed = seed)

# set the parameters that dictate output
print_and_plot = true                    # Print out useful info about sim and plot simulation time series info
write = true                      # Says whether or not the messages data must be written to a file
i = 1
# run the simulation
try 
    StartLOB(gateway)
    @time simulate(parameters, gateway, print_and_plot, write)
    # @time s(gateway, i)
    EndLOB(gateway)
catch e

    @error "Something went wrong" exception=(e, catch_backtrace())

finally

    # Close up resources even if an error has occured
    EndLOB(gateway)

    Logout(gateway)

    # StopCoinTossX()

end