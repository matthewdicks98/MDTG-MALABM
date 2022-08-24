# Simple learning agents interacting with an agent-based market model

We consider the learning dynamics of a single reinforcement learning optimal execution trading agent when it interacts with an event driven agent-based financial market model. Trading takes place asynchronously through a matching engine in event time. The optimal execution agent is considered at different levels of initial order-sizes and differently sized state spaces. The resulting impact on the agent-based model and market are considered using a calibration approach that explores changes in the empirical stylised facts and price impact curves. Convergence, volume trajectory and action trace plots are used to visualise the learning dynamics. The event-time ABM is an extension of the hybrid stochastic time ABM with the description found [here](https://arxiv.org/abs/2108.07806) and the implementation found in this repository: [IJPCTG-ABMCoinTossX](https://github.com/IvanJericevich/IJPCTG-ABMCoinTossX).

## Prerequisites
---
1. Julia version 1.7.1.
2. A text editor such as [VSCode](https://code.visualstudio.com/) or [Atom](https://atom.io/).
4. [Git](https://git-scm.com/)
3. The matching engine used is a fork from the official release of version [v1.1.0](https://github.com/dharmeshsing/CoinTossX/tree/v1.1.0) of [CoinTossX](https://github.com/dharmeshsing/CoinTossX). This fork was used because it contains a UDP socket on which market data updates can be received from the matching engine. The fork can be found [here](https://github.com/IvanJericevich/CoinTossX).<Need to update to include the wait loop removal.>

## Usage 
---

To clone this repository:

```console
git clone https://github.com/matthewdicks98/MDTG-MALABM.git
```

Julia packages can be installed from the Julia REPL using:

```console
using Pkg
Pkg.add(" ")
```

### Branch structure

Given that there are multiple stages to the project it was decided that each stage will be published on its own branch. The [Calibrated-ABM](https://github.com/matthewdicks98/MDTG-MALABM/tree/Calibrated-ABM) branch contains all the files needed to:

* Simulate the event-based ABM.
* Perform the sensitivity analysis and calibration of the ABM.
* Clean JSE L1LOB TAQ data as well as clean raw message data from the ABM simulation.
* Generate price path and stylised fact visualisations.

The [RL-ABM](https://github.com/matthewdicks98/MDTG-MALABM/tree/RL-ABM) branch extends the [Calibrated-ABM](https://github.com/matthewdicks98/MDTG-MALABM/tree/Calibrated-ABM) branch's functionality and contains additional files needed to:

* Simulate the event-based ABM with an RL agent included for one episode.
* Train an RL agent inside the ABM.
* Visualise the convergence and learning dynamics of the RL agent.
* Visualise the effect the RL agent had on the ABM.

Once the repository has been cloned you can switch between branches using:

```console
git checkout <branch_name>
```

## Reproducibility
---

### Key files

On the [Calibrated-ABM](https://github.com/matthewdicks98/MDTG-MALABM/tree/Calibrated-ABM) branch:

* [RunReactiveABM.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/RunReactiveABM.jl) runs the event-based simulation once for a given set of parameters.  
* [SensitivityAnalysis.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/SensitivityAnalysis.jl) runs the sensitivity analysis for given ranges of parameter values.
* [Calibration.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/Calibration.jl) uses [NMTA.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/NMTA.jl) to calibrate the ABM to a day of JSE data. [NMTA.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/NMTA.jl) implements the Nelder-Mead with threshold accepting optimisation algorithm.
* [CoinTossX.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/DataCleaning/CoinTossX.jl) and [JSE.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/DataCleaning/JSE.jl) clean the ABM and JSE data into the L1LOB format respectively. [CoinTossX.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/DataCleaning/CoinTossX.jl) also visualises the realised price path.
* [StylisedFacts.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/Calibrated-ABM/Scripts/StylisedFacts.jl) visualises the stylised facts found in the ABM and the JSE.

In addition to the functionality listed above the [RL-ABM](https://github.com/matthewdicks98/MDTG-MALABM/tree/RL-ABM) branch contains:

* [RLUtilities.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/RL-ABM/Scripts/RLUtilities.jl) which contains functions that can simulate an RL agent in an ABM for one episode and train the RL agent to optimally liquidate a parent order. 
* [RLVisualizations.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/RL-ABM/Scripts/RLVisualizations.jl) which allows you to visualise different aspects of the RL agent's convergence as well as the learning dynamics and how the agent has affected the ABM.
* The [StylisedFacts.jl](https://github.com/matthewdicks98/MDTG-MALABM/blob/RL-ABM/Scripts/StylisedFacts.jl) file includes extra functionality that allows you to visualise stylised facts for the RL agent.

<Data structure To replicate the data analysis the following file paths hold the required information: Data>

## Authors
---
1. Matthew Dicks
2. Tim Gebbie

## Publications
---

The link to the arxiv paper can be found [here](https://arxiv.org/abs/2208.10434).

## References
---
Since this work extends previous work code was used from the repository [IJPCTG-ABMCoinTossX](https://github.com/IvanJericevich/IJPCTG-ABMCoinTossX). 