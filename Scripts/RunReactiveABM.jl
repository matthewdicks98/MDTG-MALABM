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
Nᴸₜ = 5
Nᴸᵥ = 5
Nᴴ = 30
δ = 0.01
κ = 10
ν = 3.3
m₀ = 10000
σᵥ = 0.01
σₜ = 0.000
λmin = 0.0005
λmax = 0.03
γ = Millisecond(800)
T = Millisecond(8000) # an hour is 3600 * 1000
seed = 2

parameters = Parameters(Nᴸₜ = Nᴸₜ, Nᴸᵥ = Nᴸᵥ, Nᴴ = Nᴴ, δ = δ, κ = κ, ν = ν, m₀ = m₀, σᵥ = σᵥ, σₜ = σₜ, λmin = λmin, λmax = λmax, γ = γ, T = T, seed = seed)

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