#=
Moments:
- Julia version: 1.5.3
- Authors: Ivan Jericevich, Patrick Chang, Tim Gebbie
- Function: Compute moments of mid-price time-series
- Structure:
    1. Moments structure
    2. Hurst exponent
    3. GPH estimator
    4. Hill estimator
=#
import HypothesisTests: ADFTest, ApproximateTwoSampleKSTest
import Statistics: mean, quantile, std
import StatsBase.kurtosis
import GLM: lm, coef
using ARCHModels, Polynomials
#---------------------------------------------------------------------------------------------------

#----- Moments structure -----#
struct Moments # Moments of log-returns
    μ::Float64 # Mean
    σ::Float64 # Standard deviation
    κ::Float64 # Kurtosis
    ks::Float64 # Kolmogorov-Smirnov test statistic for the difference between distributions
    hurst::Float64 # Hurst exponent: hurst < 0.5 => mean reverting; hurst == 0.5 => random walk; hurst > 0.5 => momentum
    gph::Float64 # GPH estimator representing long-range dependence
    adf::Float64 # ADF statistic representing random walk property of returns
    garch::Float64 # GARCH paramaters representing short-range dependence
    hill::Float64 # Hill estimator
    function Moments(logreturns1::Vector{Float64}, logreturns2::Vector{Float64})
        μ = round(mean(logreturns1), digits = 3); σ = round(std(logreturns1), digits = 3); κ = round(kurtosis(logreturns1), digits = 3)
        if logreturns1 == logreturns2 # due to ties we had that sometimes the ks stat would reject H₀ when logreturns1 == logreturns2
            ks = 0
        else
            ks = round(ApproximateTwoSampleKSTest((logreturns1), (logreturns2)).δ, digits = 3)
        end
        hurst = round(HurstExponent(logreturns1), digits = 3)
        gph = round(GPH(abs.(logreturns1)), digits = 3)
        adf = round(ADFTest(logreturns1, :none, 0).stat, digits = 3)
        garch = round(sum(coef(ARCHModels.fit(GARCH{1, 1}, logreturns1))[2:3]), digits = 3)
        hill = round(HillEstimator(logreturns1[findall(x -> (x >= quantile(logreturns1, 0.95)) && (x > 0), logreturns1)], 50), digits = 3)
        new(μ, σ, κ, ks, hurst, gph, adf, garch, hill)
    end
end
#---------------------------------------------------------------------------------------------------

#----- Hurst exponent -----#
function HurstExponent(x, d = 100)
    N = length(x)
    if mod(N, 2) != 0 x = push!(x, (x[N - 1] + x[N]) / 2); N += 1 end
    N₁ = N₀ = min(floor(0.99 * N), N-1); dv = Divisors(N₁, d)
    for i in (N₀ + 1):N
        dw = Divisors(i, d)
        if length(dw) > length(dv) N₁ = i; dv = copy(dw) end
    end
    OptN = Int(N₁); d = dv
    x = x[1:OptN]
    RSempirical = map(i -> RS(x, i), d)
    return coeffs(Polynomials.fit(Polynomial, log10.(d), log10.(RSempirical), 1))[2] # Hurst is slope of log-log linear fit
end
function Divisors(n, n₀)
    return filter(x -> mod(n, x) == 0, n₀:floor(n/2))
end
function RS(z, n)
    y = reshape(z, (Int(n), Int(length(z) / n)))
    μ = mean(y, dims = 1)
    σ = std(y, dims = 1)
    temp = cumsum(y .- μ, dims = 1)
    return mean((maximum(temp, dims = 1) - minimum(temp, dims = 1)) / σ)
end
#---------------------------------------------------------------------------------------------------

#----- GPH estimator -----#
function GPH(x, bandwidthExponent = 0.5)
    n = length(x); g = Int(trunc(n^bandwidthExponent))
    j = 1:g; kk = 1:(n - 1)
    w = 2 .* π .* j ./ n # x .-= mean(x)
    σ = sum(x .^ 2) / n
    Σ = map(k -> sum(x[1:(n - k)] .* x[(1 + k):n]) / n, kk)
    periodogram = map(i -> σ + 2 * sum(Σ .* cos.(w[i] .* kk)), j)
    indeces = j[findall(x -> x > 0, periodogram)]
    x_reg = 2 .* log.(2 .* sin.(w[indeces] ./ 2)); y_reg = log.(periodogram[indeces] ./ (2 * π))
    regression = lm(hcat(ones(length(x_reg)), x_reg), y_reg)
    return abs(coef(regression)[2])
end
#---------------------------------------------------------------------------------------------------

#----- Hill estimator -----#
function HillEstimator(x, iterations)
    N = length(x)
    logx = log.(x)
    L = minimum(x); R = maximum(x)
    α = 1 / ((sum(logx) / N) - log(L))
    for i in 1:iterations
        C = (log(L) * (L^(-α)) - log(R) * (R^(-α))) / ((L^(-α)) - (R^(-α)))
        D = (R^α * L^α * (log(L) - log(R))^2) / (L^α - R^α)^2
        α = α * (1 + (α * (sum(logx) / N) - α * C - 1) / (α^2 * D - 1))
    end
    return α
end
#---------------------------------------------------------------------------------------------------