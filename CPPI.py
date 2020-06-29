import numpy as np
import math
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
from random import choice
from string import ascii_lowercase
from statistics import stdev
from statistics import mean
from Heston import Gen_var_t
from Heston import lnX_t

# Project: CPPI Portfolios Simulations under Different Risky Assets
# By: Victor Felipe Gontijo - Quantitative research intern - BNP Paribas - Paris
# Proposed by: Jean-Philippe Lemor - Head of systematic strategies and hybrids quantitative research team - BNP Paribas - Paris
# June 2020

# We define here, all functions that are used in the project:

# Plots the graph of a process
def plot(title, y, dt=1, color_ = 'green', name_ = 'fig'+ ''.join(choice(ascii_lowercase) for i in range(5)) ):
    
    l= len(y)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.plot(instants, y, color= color_, linestyle='dashed', linewidth = 0.1, marker='o', markersize=0.5)
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.savefig(name_)

# Plots and superimpose the graph of three process
def plot_superimpose(title, name, y1, y2, y3, dt=1):

    l= len(y1)
    instants = np.array(range(0,l))*dt
    
    plt.figure(figsize=(15,8))

    plt.scatter(instants, y1, s=2, color='red')
    plt.scatter(instants, y2, s=2, color='green')
    plt.scatter(instants, y3, s=2, color='blue')
   
    plt.xlabel('Instants (Years)') 
    plt.ylabel('Value') 
  
    plt.title(title)
    plt.savefig(name)

# Generates a Geometric Brownian Motion path 
def Geometric_Brownian_Motion(S_0, mu, sigma, instants):
    
    l = len(instants)
    
    positions = [S_0]

    # Generate the increments
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        positions.append( positions[i-1]*math.exp( (mu - sigma*sigma/2 )*dt + sigma*np.random.normal(0, math.sqrt(dt)) ) )

    return positions

# Generates the path of a riskless asset 
def Riskless_Returns(r, Rl_0, instants):

    l = len(instants)
    
    positions = [Rl_0]

    # Generates the increments of the process
    for i in range(1,l):
        dt = instants[i] - instants[i-1]
        positions.append( positions[i-1]*math.exp(r*dt) )

    return positions

# Generates a path of the stochastic variance, following the Heston's Model dynamics
def Heston_var(instants, V_0, kappa, theta, xi, rho):
        
    l = len(instants)
    
    positions = [V_0]

    # Generates the increments of the process
    for i in range(1,l):
        delta = instants[i] - instants[i-1]
        positions.append( Gen_var_t(delta, kappa, theta, xi, positions[i-1]) )

    return positions

# Generates a path of risky assets prices, following the Heston's Model dynamics
def Heston_X(r, instants, var, S_0, kappa, theta, xi, rho, gamma1 = 0.5, gamma2 = 0.5):
    
    l = len(instants)
    
    positions = [math.log(S_0)]

    # Generates the increments of the process
    for i in range(1,l):
        delta = instants[i] - instants[i-1]
        positions.append( lnX_t(r, var[i-1], var[i], positions[i-1], delta, kappa, theta, xi, rho, gamma1, gamma2) ) 

    return np.exp(positions)

# Calculates the real values of a CPPI Portfolio at the rebalancing dates 
def CPPI_PortfolioValue (m, rebalancing_instants, risky_prices, V_0, F_0, r):

    l = len(rebalancing_instants)
    
    portfolio_value = [V_0]
    floor = F_0

    risky_position = m*(V_0 - floor)
    riskless_position = V_0 - risky_position
    

    for i in range(1,l):
        dt = rebalancing_instants[i] - rebalancing_instants[i-1]
            
        #Ratio of the assets' prices between the current rebalancing instant and the last one
        ratio_riskless = math.exp(r*dt)
        ratio_risky = risky_prices[i]/risky_prices[i-1]
            
        # New value of the portfolio
        V_t = risky_position*ratio_risky + riskless_position*ratio_riskless
        portfolio_value.append(V_t)
            
        #New value of the floor
        floor = floor*ratio_riskless
            
        risky_position = m*(portfolio_value[i] - floor)
        riskless_position = portfolio_value[i] - risky_position
    
    return portfolio_value

# Calculates, via Equation 9, approximated values of a CPPI Portfolio at the rebalancing dates
# Designed to the context in which the Risky Asset follows a Geometric Brownian Motion
def Gbm_Formula_CPPI_PortfolioValue(m, rebalancing_instants, risky_prices, V_0, F_0, r, sigma):
    
    l = len(rebalancing_instants)
    
    floor = F_0
    beta = r*(1-m) + 0.5*m*(1 -m)*sigma*sigma
    alpha = (V_0 - F_0)/pow(risky_prices[0],m)

    portfolio_value =  [V_0]
    
    for i in range(1,l):
        
        dt = rebalancing_instants[i] - rebalancing_instants[i-1]
        
        ratio_floor = math.exp(r*dt)
        ratio_alpha = math.exp(beta*dt)

        floor = floor*ratio_floor
        alpha = alpha*ratio_alpha

        c_t = alpha*pow(risky_prices[i],m)

        portfolio_value.append(floor + c_t)
    
    return portfolio_value

# Calculates, via Equation 9, approximated values of a CPPI Portfolio at the rebalancing dates
# Designed to the context in which the Risky Asset follows a Heston Model
def Heston_Formula_CPPI_PortfolioValue(m, rebalancing_instants, risky_prices, var, V_0, F_0, r):
    
    l = len(rebalancing_instants)
    
    sum_sup = 0
    sum_inf = 0
    integ = (sum_sup + sum_inf)/2
    sum_dt = 0
    
    floor = F_0
    alpha_0 = (V_0 - F_0)/pow(risky_prices[0],m)

    portfolio_value =  [V_0]
    
    for i in range(1,l):
        
        dt = rebalancing_instants[i] - rebalancing_instants[i-1]

        sum_dt = sum_dt + dt
        sum_inf = sum_inf + var[i-1]*dt
        sum_sup = sum_sup + var[i]*dt
        integ = (sum_sup + sum_inf)/2

        beta = r*(1-m) + 0.5*m*(1 -m)*integ/sum_dt
        
        ratio_floor = math.exp(r*dt)
        
        floor = floor*ratio_floor
        alpha = alpha_0*math.exp(beta*sum_dt)

        c_t = alpha*pow(risky_prices[i],m)

        portfolio_value.append(floor + c_t)
    
    return portfolio_value

# Generates several paths of a CPPI portfolio based in Geometric Brownian Motion Risky Asset
# For each single simulated path, the values of the CPPI portfolio are determined by two methods: Exactly and Approximated
# For each single simulated path, it calculates a "distance" beetween the two results 
def Distances_GenerateSampleBM (sample_size, m, instants, S_0, V_0, F_0, r, mu, sigma, norm = 'l2'):

    distances_vector = []

    for i in range (0, sample_size):

        risky_prices = Geometric_Brownian_Motion(S_0, mu, sigma, instants)

        pv = np.array( CPPI_PortfolioValue (m, instants, risky_prices, V_0, F_0, r) )
        fpv = np.array ( Gbm_Formula_CPPI_PortfolioValue(m, instants, risky_prices, V_0, F_0, r, sigma) )
        
        if(norm == 'l2'):
            distances_vector.append( np.linalg.norm(pv-fpv) )
        
        else:
            distances_vector.append( np.linalg.norm( np.divide( pv-fpv, pv), np.inf) )
    
    return distances_vector

# Generates several paths of a CPPI portfolio based in Heston's Model Risky Asset
# For each single simulated path, the values of the CPPI portfolio are determined by two methods: Exactly and Approximated
# For each single simulated path, it calculates a "distance" beetween the two obtained results 
def Distances_GenerateSampleH (sample_size, m, instants, S_0, V_0, F_0, var_0, r, kappa, theta, xi, rho, norm = 'l2'):

    distances_vector = []

    for i in range (0, sample_size):

        var = Heston_var(instants, var_0, kappa, theta, xi, rho)
        risky_prices = Heston_X(r, instants, var, S_0, kappa, theta, xi, rho, 0.5, 0.5)

        pv = np.array( CPPI_PortfolioValue (m, instants, risky_prices, V_0, F_0, r) )
        fpv = np.array ( Heston_Formula_CPPI_PortfolioValue(m, instants, risky_prices, var, V_0, F_0, r) )
        
        if(norm == 'l2'):
            distances_vector.append( np.linalg.norm(pv-fpv) )
        
        else:
            distances_vector.append( np.linalg.norm( np.divide( pv-fpv, pv), np.inf) )
    
    return distances_vector

# Generates statics about the sample of "distances", calculated with the functions above 
def Distances_Statistcs (sample, n_instants, norm = 'l2', name_ = 'fig'+ ''.join(choice(ascii_lowercase) for i in range(5))):

    sp_size = len(sample)
    sp_mean = mean(sample)
    sp_stdev = stdev(sample)

    out = 0
    for x in sample:
        if abs(x - sp_mean) > sp_stdev:
            out = out + 1

    out_prop = out/sp_size

    print()
    if(norm == 'l2'):
        print ('Euclidean Distance - dim: ' + str(n_instants) +' - l2 norm of f - g')
    else:
        print ('Supremum norm of relative differences: linf norm of h')

    print ('Number of rebalancing dates: ' + str(n_instants))
    print ('Mean of Distances Sample: ' + str(sp_mean))
    print ('Standard Deviation of Distances Sample: ' + str(sp_stdev))
    print ('Proportion of realisations that have distance from the mean greater than the standard deviation: ' + str(out_prop))

    plt.figure(figsize=(20,10))
    plt.hist(x=sample, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)

    if(norm == 'l2'):
        plt.xlabel('Euclidean distance between the two results - dimension: ' + str(n_instants))
    else:
        plt.xlabel('Supremum norm of relative differences between the two results - dimension: ' + str(n_instants))
           
    plt.ylabel('Frequency')
    plt.title('Distance Histogram: '+ str(sp_size) + ' realisations')
    plt.savefig(name_)




