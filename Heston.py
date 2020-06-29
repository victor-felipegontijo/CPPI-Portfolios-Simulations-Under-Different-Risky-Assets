import math
import numpy as np
from scipy.stats import ncx2 

# We define here, all functions that are necessary to simulate the returns of a risky asset following the Heston Model
# This implementation is based on the article: A Fast and Exact Simulation for CIR Process, by Anqi Shao

# Parameters to generate an increment of the stochastic variance following the Heston's model dynamics
def c(delta, kappa, xi):
    return xi*xi*(1 - math.exp(-kappa*delta))/(4*kappa)

def d(kappa, theta, xi):
    return 4*kappa*theta/(xi*xi)

def lbd(delta, kappa, xi, var_s):
    return var_s*math.exp(-kappa*delta)/c(delta, kappa, xi)

# Generate an increment of the stochastic variance following the Heston's model dynamics
# Suppose we know the variance at the instant s, it gives a realisation of the variance at the instant t = s + delta  
def Gen_var_t(delta, kappa, theta, xi, var_s):
    return c(delta, kappa, xi)*ncx2.rvs( d(kappa, theta, xi), lbd(delta, kappa, xi, var_s) )

# Parameters to generate an increment of the risky asset price following the Heston's model dynamics
def K_0(r, delta, kappa, theta, rho, xi):
    return (r - rho*kappa*theta/xi)*delta

def K_1(delta, kappa, xi, rho, gamma1):
    return gamma1*delta*( kappa*rho/xi - 0.5) - rho/xi

def K_2(delta, kappa, xi, rho, gamma2):
    return gamma2*delta*( kappa*rho/xi - 0.5) + rho/xi

def K_3(delta, rho, gamma1):
    return gamma1*delta*(1 - rho*rho)

def K_4(delta, rho, gamma2):
    return gamma2*delta*(1 - rho*rho)

# Generate an increment of ln-prices of a risky asset, following the Heston's model dynamics
# Suppose we know the ln-price of the asset, at the instant s, it gives a realisations of ln-price of the asset, at the instant t = s + delta 
def lnX_t(r, var_s, var_t, lnX_s, delta, kappa, theta, xi, rho, gamma1, gamma2):
        
    z = np.random.normal(0, 1)
        
    s1 = lnX_s 
    s2 = K_0(r, delta, kappa, theta, rho, xi)
    s3 = K_1(delta, kappa, xi, rho, gamma1)*var_s
    s4 = K_2(delta, kappa, xi, rho, gamma2)*var_t
    s5 = K_3(delta, rho, gamma1)*var_s
    s6 = K_4(delta, rho, gamma2)*var_t

    return s1 + s2 + s3 + s4 + math.sqrt(s5 + s6)*z
    
