ENV["JULIA_COPY_STACKS"]=1
path_to_files = "/home/matt/Desktop/Advanced_Analytics/Dissertation/Code/MDTG-MALABM/"
include(path_to_files * "Scripts/ReactiveABM.jl") # also includes CoinTossXUtilities.jl

# start CTX, start the JVM and 
# StartCoinTossX(build = false, deploy = true) # always redeploy to free up memory after last simulation
# sleep(5)

StartJVM()

gateway = Login(1,1)

StartLOB(gateway)

# set the parameters
Nᴸₜ = 6
Nᴸᵥ = 4
Nᴴ = 30
δ = 0.01
κ = 5
ν = 3.3
m₀ = 10000
σᵥ = 0.01
λmin = 0.0005
λmax = 0.05
γ = Millisecond(1000)
T = Millisecond(30000) # an hour is 3600 * 1000
seed = 100 # 125 has price decrease

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, λmin = λmin, λmax = λmax, γ = γ, T = T, seed = seed)

# set the parameters that dictate output
print_and_plot = true                    # Print out useful info about sim and plot simulation time series info
write = true                      # Says whether or not the messages data must be written to a file

# run the simulation
try 

    @time simulate(parameters, gateway, print_and_plot, write)

catch e

    @error "Something went wrong" exception=(e, catch_backtrace())

finally

    # Close up resources even if an error has occured
    EndLOB(gateway)

    Logout(gateway)

    # StopCoinTossX()

end