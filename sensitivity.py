import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel,delayed
from friendlyNet import *

def compute_j(x,net):
    j = (net.T*x).T
    np.fill_diagonal(j,0)
    j = j + np.diag(1 + np.dot(net,x) + np.diag(net)*x)
    return j

def sense_kl(t,ps,x,k,l,net):
    return x(t)[k]*x(t)[l]*np.eye(net.shape[0])[k] + np.dot(compute_j(x(t),net),ps)

def get_sensitivity_single_trajectory(fnet,i,k,l):
    soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),1000)
    sensitivity = solve_ivp(sense_kl,(0,soln.t[-1]),np.zeros(fnet.Adjacency.shape[0]),args = (soln.sol,k,l,fnet.Adjacency.T))
    return np.mean(sensitivity.y[i])

def get_sensitivity(target_node,fnet,entry,numtrials = 10000,nj=-1):
    k = fnet.NodeNames.index(entry[0])
    l = fnet.NodeNames.index(entry[1])
    i = fnet.NodeNames.index(target_node)
    return np.mean(Parallel(n_jobs = nj)(delayed(get_sensitivity_single_trajectory)(fnet,i,k,l) for tri in range(numtrials)))

def sense_all(t,p,x,net):
    N= net.shape[0]
    all_together = np.concatenate([sense_kl(t,p[(N**2)*l + N*k:(N**2)*l + (k+1)*N],x,k,l,net) for k in range(N) for l in range(N)])
    return all_together

def get_all_sensitivity_st(fnet,i,mxTime = 1000,shift=0):
    y0 = np.zeros(fnet.Adjacency.shape[0]**3)
    soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift=shift)
    all_sensitivity = solve_ivp(sense_all,(0,soln.t[-1]),y0,args = (soln.sol,fnet.Adjacency.T-shift)).y
    #Take weighted average over trajectory, weighting towards end
    wbase = 2
    weights = np.array([wbase**i for i in range(all_sensitivity.shape[1])])
    weights = weights/sum(weights)
    weighted_avg = np.dot(all_sensitivity,weights)
    param_senses = weighted_avg.reshape(fnet.Adjacency.shape[0],fnet.Adjacency.shape[0]**2)[i].reshape(fnet.Adjacency.shape).T
    return param_senses

def get_all_sensitivity(target_node,fnet,numtrials = 1000,mxTime=1000,nj = -1,shift=0):
    i = fnet.NodeNames.index(target_node)
    mnsense = sum(Parallel(n_jobs = nj)(delayed(get_all_sensitivity_st)(fnet,i,mxTime=mxTime,shift=shift) for tri in range(numtrials)))/numtrials
    sensitivities = pd.DataFrame(mnsense.T, columns = fnet.NodeNames,index = fnet.NodeNames)
    return sensitivities
