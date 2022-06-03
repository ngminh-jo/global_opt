"""
implementation of Frank-Wolfe algorithm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


# import data
shares = ['AAPL', 'MSFT', 'XOM', 'JNJ', 'GE', 'BRK-B', 'FB',
          'AMZN', 'WFC','T','GOOGL','PG','GOOG','VZ','PFE',
          'CVX','KO','HD','DIS','BAC','MRK','V','PM','CMCSA',
          'INTC','PEP','CSCO','C','GILD','IBM']
historical_prices = pd.read_excel('historical_prices.xlsx', index_col=0)
historical_returns = historical_prices / historical_prices.shift(1) - 1
# why drop??
# historical_returns = historical_returns.drop(['13.05.2015  00:00:00'])

# input
#######
historical_returns = historical_returns.iloc[1:]
global mu
mu = historical_returns.mean()*252 # Estimation of mu
global sigma
sigma = historical_returns.cov()*252 # Estimation of covariance-matrix
sigma = sigma.values
mu = mu.values
global lmda
lmda = 10
global n
n = mu.size


# function to be minimized
def fun(x):
    return np.matmul(-mu.T, x)+0.5*lmda*np.matmul(x.T, np.matmul(sigma, x))

# parameters of Frank-Wolfe algorithm
x0 = np.full(n, 1/n)
print("x0: ", x0)
epsilon = 0.0001
# input end
###########

# helper functions for main
def f_inner(x, *args):
    xk = args[0]
    grad_xk = -mu + lmda*np.matmul(sigma, xk)
    return np.inner(grad_xk, x-xk)

def f_t(t, *args):
    xk = args[0]
    dk = args[1]
    return fun(xk + t*dk)

def Qk(xk):
    method = "trust-constr"
    x_guess = np.roll(xk, n//2)
    args = (xk,)
    bounds = ((0,1),)*xk.shape[0]
    lin_constraint = LinearConstraint(np.ones(xk.shape[0]), 1, 1)
    opt_res = minimize(f_inner, x_guess, args=args, method=method, bounds=bounds, constraints=lin_constraint)
    yk, vk = opt_res.x, opt_res.fun
    return yk, vk

# main Frank Wolfe loop
def fw(fun, x0, epsilon, mu=mu, lmda=lmda, sigma=sigma):
    # k = 0
    xk = np.copy(x0)
    yk, vk = Qk(xk)
    while vk<-epsilon:
        dk = yk-xk
        t_guess = 0.5
        args = (xk, dk)
        method = "Powell"
        bounds = ((0,1),)
        tk =  minimize(f_t, t_guess, args=args, method=method, bounds=bounds).x
        # now k = k+1
        xk += tk*dk
        yk, vk = Qk(xk)
        print("vk: ", vk)
    return xk


if __name__ == "__main__":
    xk = fw(fun, x0, epsilon)
    print("xk: ", xk)
