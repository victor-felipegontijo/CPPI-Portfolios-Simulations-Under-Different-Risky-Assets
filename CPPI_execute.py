import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
from CPPI import plot
from CPPI import plot_superimpose
from CPPI import Geometric_Brownian_Motion as Gbm 
from CPPI import Riskless_Returns as Rls
from CPPI import CPPI_PortfolioValue as Pv
from CPPI import Gbm_Formula_CPPI_PortfolioValue as GFpv
from CPPI import Heston_Formula_CPPI_PortfolioValue as HFpv
from CPPI import Heston_var
from CPPI import Heston_X
from CPPI import Distances_GenerateSampleBM as DGSample
from CPPI import Distances_GenerateSampleH as DGHSample
from CPPI import Distances_Statistcs as DStats

# Project: CPPI Portfolios Simulations under Different Risky Assets
# By: Victor Felipe Gontijo - Quantitative research intern - BNP Paribas - Paris
# Proposed by: Jean-Philippe Lemor - Head of systematic strategies and hybrids quantitative research team - BNP Paribas - Paris
# June 2020

# We execute here the simulations proposed along the project:
# These simulations are also available in the notebook "CPPI.ipynb". There, they are well-explained and presented in the logical context of the project.
# We recomend the utilisation of the notebook for a whole comprehension of the project.

#___________________________________________#

## 1 - General parameters of investment

# 1.1 - Years of investment
years = 5

# 1.2 - Annual interest rate
r = 0.1

# 1.3 - Initial capital
V_0 = 2.5

#___________________________________________#

## 2 - CPPI strategy parameters

# 2.1 - Number of times the portfolio is rebalanced per day
rebalancings_per_day = 10

# 2.2 - Initial floor
F_0 = 1

# 2.3 - Multiple
m = 3

#___________________________________________#

# Code parameters
dt = 1/(365*rebalancings_per_day)
size = int(years/dt) + 1
rebalacing_instants = np.array(range(0,size))*dt

#___________________________________________#

#### PATH SIMULATIONS 


### S1) *Risky asset's* returns following a Geometric Brownian Motion (log-normal distribution)

# S1.1 - Risky asset's returns parameters

# S1.1.1 - Drift
mu = 0.04

# S1.1.2 - Volatility
sigma = 0.2

# S1.1.3 - Initial value of the risky assets
S_0 = 2.5

## S1.2 - Simulating a Geometric Brownian Motion path of the *risky asset's* prices over the chosen period
S = Gbm(S_0, mu, sigma, rebalacing_instants)
plot('Log-normal Risky Asset path:', S, dt, 'red', 'Log-normal Risky Asset path')

## S1.3 - Simulating the *floor's* evolution over the chosen period
F = Rls(r, F_0, rebalacing_instants)
plot('Floor evolution', F, dt, 'blue', 'Floor')

## S1.4 - Determining the CPPI portfolio real value over the simulated Geometric Brownian Motion path
V = Pv(m, rebalacing_instants, S, V_0, F_0, r)
plot('Log-normal Risky Asset - CPPI Portfolio value evolution', V, dt, 'green', 'Portfolio Value')

## S1.5 - Comparison between the *Risky Asset*, the *CPPI portfolio* and the *floor*, over the simulated Geometric Brownian Motion path
plot_superimpose('Log-normal Risky Asset x CPPI Portfolio value x Floor evolution', 'Log-normal Risky Asset x Portfolio value x Floor', S , V, F, dt)

## S1.6 - Detemining the values of the *CPPI portfolio* calculated by the  *Equation 9* over the simulated path
GFV = GFpv(m, rebalacing_instants, S, V_0, F_0, r, sigma)
plot('Log-normal Risky asset - CPPI Portfolio value calculated by formula (9)', GFV, dt, 'black', 'Formula(9)- Log-normal Risky asset - Portfolio Value')


#### Analyzing the difference between the real values and the values provided by Equation 9

### A1) *Risky asset's* returns following a Geometric Brownian Motion (log-normal distribution)

## A1.1 - Parameters of the sample of differences

# A1.1.1 Number of paths
Samples_size = 100

# A1.1.2 Number of times the portfolio is rebalanced per day
rebalancings_per_day = 1

# Code parameters
dt = 1/(365*rebalancings_per_day)
size = int(years/dt) + 1
rebalacing_instants = np.array(range(0,size))*dt

## A1.2 - Sample of distances : l2 norm of absolute differences
l2_Sample1 = DGSample(Samples_size, m, rebalacing_instants, S_0, V_0, F_0, r, mu, sigma, 'l2')
DStats(l2_Sample1, size, 'l2', 'Sample of Distances - l2')

## A1.3 - Sample of distances: sup norm of relative differences
linf_Sample1 = DGSample(Samples_size, m, rebalacing_instants, S_0, V_0, F_0, r, mu, sigma, 'linf')
DStats(linf_Sample1, size, 'linf', 'Sample of Distances - linf')


### S2) *Risky asset's* returns following a Heston Model

## S2.1 - *Risky asset's* and Stochastic Variance parameters

# S2.1.1 Drift
mu = 0.08

# S2.1.2 Long term variance
theta = 0.04

# S2.1.3 Volatility of volatility
xi = 0.3

# S2.1.4 Correlation between Brownian Motions
rho = -0.3

# S2.1.5 Returning to the mean speed
kappa = 1

# S2.1.6 Initial variance
var_0 = 0.04

# S2.1.7 Initial value of the risky assets
S_0 = 2.5

# Code Parameters
rebalancings_per_day = 10
dt = 1/(365*rebalancings_per_day)
size = int(years/dt) + 1
rebalacing_instants = np.array(range(0,size))*dt

## S2.2 - Simulating a Heston  path of variances over the chosen period
var = Heston_var(rebalacing_instants, var_0, kappa, theta, xi, rho)
plot('Heston Variances Path', var, dt, 'black', 'Heston Variances Path')

## S2.3 - Simulating a Heston path of *risky asset's* returns over the chosen period
X = Heston_X(mu, rebalacing_instants, var, S_0, kappa, theta, xi, rho, 0.5, 0.5)
plot('Heston Risky Asset Path', X, dt, 'red', 'Heston risky path')

## S2.4 - Simulating the *floor's* evolution over the chosen period
F = Rls(r, F_0, rebalacing_instants)
plot('Floor evolution', F, dt, 'blue', 'Floor')

## S2.5 - Determining the CPPI portfolio real value over the simulated Heston path
HV = Pv(m, rebalacing_instants, X, V_0, F_0, r)
plot('Heston Risky Asset - CPPI Portfolio value evolution', HV, dt, 'green', 'Heston - CPPI Portfolio Value')

## S2.6 - Comparison between the *Risky Asset*, the *CPPI portfolio* and the *floor*, over the simulated Heston path
plot_superimpose('Heston-Risky Asset evolution x CPPI Portfolio value x Floor evolution', 'Heston-Risky Asset x Portfolio value x Floor', X , HV, F, dt)

## S2.7 - Detemining the values of the *CPPI portfolio* calculated by the *Equation 9* over the Heston simulated path
HFV = HFpv(m, rebalacing_instants, X, var, V_0, F_0, r)
plot('Heston-Risky Asset - CPPI Portfolio value calculated by formula (9)', HFV, dt, 'black', 'Formula(9)-Heston-Portfolio Value')


#### Analyzing the difference between the real values and the values provided by Equation 9

### A2) *Risky asset's* returns following a Heston Model

## A2.1 - Parameters of the sample of differences

# A1.2.1 Number of paths
Samples_size = 100

# A1.1.2 Number of times the portfolio is rebalanced per day
rebalancings_per_day = 1

# Code parameters
dt = 1/(365*rebalancings_per_day)
size = int(years/dt) + 1
rebalacing_instants = np.array(range(0,size))*dt

## A2.2 - Sample of distances : l2 norm of absolute differences
l2_SampleH = DGHSample(Samples_size, m, rebalacing_instants, S_0, V_0, F_0, var_0, r, kappa, theta, xi, rho, 'l2')
DStats(l2_SampleH, size, 'l2', 'Sample of Distances - l2')

## A2.3 - Sample of distances: sup norm of relative differences
linf_SampleH = DGHSample(Samples_size, m, rebalacing_instants, S_0, V_0, F_0, var_0, r, kappa, theta, xi, rho, 'linf')
DStats(linf_SampleH, size, 'linf', 'Sample of Distances - linf')


plt.show()