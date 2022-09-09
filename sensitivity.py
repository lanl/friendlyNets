import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel,delayed
from friendlyNet import *

def compute_j(x,net):
    j = (net.T*x).T
    np.fill_diagonal(j,0)
    j = j + np.diag(1 + np.dot(net,x) + 2*np.diag(net)*x)
    return j

def sense_kl(t,ps,x,k,l,net):
    return x(t)[k]*x(t)[l]*np.eye(net.shape[0])[k] + np.dot(compute_j(x(t),net),ps)

def get_sensitivity_single_trajectory(fnet,i,k,l,shift = 0,weights = [],mxTime = 1000):
    soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift = shift)
    sensitivity = solve_ivp(sense_kl,(0,soln.t[-1]),np.zeros(fnet.Adjacency.shape[0]),args = (soln.sol,k,l,fnet.Adjacency.T-shift))
    if len(weights):
        weighted_avg = np.dot(sensitivity.y[i][-min(len(weights),len(sensitivity.y[i])):],weights[-min(len(weights),len(sensitivity.y[i])):])
        return weighted_avg#
    return np.mean(sensitivity.y[i])

def get_sensitivity(target_node,fnet,entry,numtrials = 100,nj=1,shift = 0,mxTime = 1000):
    k = fnet.NodeNames.index(entry[0])
    l = fnet.NodeNames.index(entry[1])
    i = fnet.NodeNames.index(target_node)
    weights = np.array([1.5**i for i in range(40)])
    weights = weights/sum(weights)
    return np.mean(Parallel(n_jobs = nj)(delayed(get_sensitivity_single_trajectory)(fnet,i,k,l,shift = shift,weights = weights,mxTime = mxTime) for tri in range(numtrials)))

def get_all_sensitivity_single_trajectory(target_node,fnet,mxTime = 1000,shift = 0,weights = []):
    all_sensitivity = np.zeros_like(fnet.Adjacency)#pd.DataFrame(columns = fnet.NodeNames,index = fnet.NodeNames)
    i = fnet.NodeNames.index(target_node)
    soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift = shift)
    for k in range(all_sensitivity.shape[0]):
        for l in range(all_sensitivity.shape[1]):
            sensitivity = solve_ivp(sense_kl,(0,soln.t[-1]),np.zeros(fnet.Adjacency.shape[0]),args = (soln.sol,k,l,fnet.Adjacency.T-shift))
            if len(weights):
                weighted_avg = np.dot(sensitivity.y[i][-min(len(weights),len(sensitivity.y[i])):],weights[-min(len(weights),len(sensitivity.y[i])):])
            else:
                weighted_avg = np.mean(sensitivity.y[i])
            all_sensitivity[k,l] = weighted_avg
            print("{}      {}          {}/{}            ".format(sensitivity.t[-1],weighted_avg,l+k*all_sensitivity.shape[0],all_sensitivity.size),end = '\r')
    return all_sensitivity

def get_all_sensitivity(target_node,fnet,mxTime = 1000,shift = 0,weights = [],nj = 1,numtrials = 100):
    mnsense = sum(Parallel(n_jobs = nj)(delayed(get_all_sensitivity_single_trajectory)(target_node,fnet,mxTime = mxTime,shift = shift,weights = weights) for tri in range(numtrials)))/numtrials
    sensitivities = pd.DataFrame(mnsense.T, columns = fnet.NodeNames,index = fnet.NodeNames)
    return sensitivities


# def sense_kl_prej(t,ps,X,k,l,prej):
#     return X[k]*X[l]*np.eye(prej.shape[0])[k] + np.dot(prej,ps)

# def sense_all(t,p,x,net):
#     N= net.shape[0]
#     prej = compute_j(x(t),net)
#     all_together = np.concatenate([sense_kl_prej(t,p[(N**2)*l + N*k:(N**2)*l + (k+1)*N],x(t),k,l,prej) for k in range(N) for l in range(N)])
#     return all_together

# def get_all_sensitivity_st(fnet,i,mxTime = 1000,shift=0,weights = []):
#     y0 = np.zeros(fnet.Adjacency.shape[0]**3)
#     soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift=shift)
#     all_sensitivity = solve_ivp(sense_all,(0,soln.t[-1]),y0,args = (soln.sol,fnet.Adjacency.T-shift))
#     #Take weighted average over trajectory, weighting towards end
#     if len(weights):
#         weighted_avg = np.dot(all_sensitivity.y[:,-min(len(weights),all_sensitivity.y.shape[1]):],weights[-min(len(weights),len(all_sensitivity.y[i])):])
#     else:
#         weighted_avg = np.mean(all_sensitivity.y,axis = 1)
#     param_senses = weighted_avg.reshape(fnet.Adjacency.shape[0],fnet.Adjacency.shape[0]**2)[i].reshape(fnet.Adjacency.shape).T
#     return param_senses

# def get_all_sensitivity(target_node,fnet,numtrials = 1000,mxTime=1000,nj = -1,shift=0):
#     i = fnet.NodeNames.index(target_node)
#     mnsense = sum(Parallel(n_jobs = nj)(delayed(get_all_sensitivity_st)(fnet,i,mxTime=mxTime,shift=shift) for tri in range(numtrials)))/numtrials
#     sensitivities = pd.DataFrame(mnsense.T, columns = fnet.NodeNames,index = fnet.NodeNames)
#     return sensitivities
