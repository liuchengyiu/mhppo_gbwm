import numpy as np
import pandas as pd
from lib.wealth_grid import WealthGrid
from lib.portfolio import Portfolio
import scipy.stats as stats
from tqdm import trange

W0 = 100
TT = 10
infusion = {0: 10, 1: 10}
cost_v = {}
grid_points = 100
port = Portfolio.get_n_portfolios()
wealth_grid = WealthGrid.gen(W0, infusion, port, cost_v, TT, grid_points)
infusions = np.zeros(TT+1)
infusions[0] = 10
infusions[1] = 10
Q = np.zeros((len(wealth_grid), TT+1, len(port)))
R = np.zeros(( len (wealth_grid), TT+1, len(port)))
H=200
h=1
alpha = 0.1
gamma = 1
epsilon = 0.3
NP = len(port)
df = pd.DataFrame(port)
print(df)
for j in range (len(wealth_grid)):
    if wealth_grid[j]>H:
        R[j,TT ,:] = 1.0

def sampleWidx(w0 ,w1 ,I,a0): #to give the next state
    mu = port[a0]['mean_variance'][0]
    sig = port[a0]['mean_variance'][1]
    p1 = stats.norm.pdf((np.log(w1/(w0+I)) -(mu -0.5* sig **2)*h)/(sig*np.sqrt(h)))
    p1 = p1/ sum (p1) #normalize probabilities
    idx = np.where(np.random.uniform() > p1.cumsum())[0]
    return len (idx) #gives index of the wealth node w1 at t1

def doOneNode(idx0 ,t0): #idx0: index on the wealth axis , t0:
    if np.random.uniform() < epsilon:
        a0 = np.random.randint(0,NP) #index of action; or plug in best
    else : 
        q = Q[idx0 ,t0 ,:]
        a0 = np.where(q==q.max ())[0]
        if len (a0) >1:
            a0 = np.random.choice(a0)
        else:
            a0 = a0[0]
    t1 = t0 + 1
    if t0 <TT: #at t<T
        w0 = wealth_grid[idx0] #scalar
        w1 = np.array(wealth_grid) #vector
        idx1 = sampleWidx(w0 ,w1 ,infusions[t0],a0)
        Q[idx0 ,t0 ,a0] = Q[idx0 ,t0,a0] + alpha *(R[idx0 ,t0,a0] + gamma*Q[idx1 ,t1 ,:]. max () - Q[idx0 ,t0,a0])
    else:
        Q[idx0 ,t0 ,a0] = (1-alpha)*Q[idx0 ,t0,a0] + alpha*R[idx0 ,t0 ,a0]
        idx1 = idx0
    return [idx1 ,t1] #gives back next state (index of W and t)

for _ in trange(100000):
    idx = wealth_grid.index(W0)
    for t in range (TT+1):
        [idx ,t] = doOneNode(idx ,t)
    