import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel,delayed
from friendlyNet import *

def compute_j(x,net):

    """
    Helper function to compute a term in the dynamical system defined by :py:func:`sense_kl <sensitivity.sense_kl>`. 

    :param x: Lotka-Volterra state at time t
    :type x: array[float]
    :param net: adjacency matrix for Lotka-Volterra system
    :type net: array[float]

    :return: term in RHS
    :rtype: array[float]
    """

    j = (net.T*x).T
    np.fill_diagonal(j,0)
    j = j + np.diag(1 + np.dot(net,x) + 2*np.diag(net)*x)
    return j

def sense_kl(t,ps,x,k,l,net):

    """
    Dynamical system for senstivity :math:`\partial x_i/\partial a_{kl}` for a Lotka-Volterra interaction parameter.

    :param t: time in simulation
    :type t: float
    :param ps: state in simulatoin
    :type ps: array[float]
    :param x: Lotka-Volterra solution (dense output of scipy.integrate.solve_ivp)
    :type x: function
    :param k: index of source of interaction
    :type k: int
    :param l: index of target of interaction
    :type l: int
    :param net: adjacency matrix for Lotka-Volterra system
    :type net: array[float]

    :return: right-hand side of dynamical system
    :rtype: array[float]
    """

    return x(t)[k]*x(t)[l]*np.eye(net.shape[0])[l] + np.dot(compute_j(x(t),net),ps)

def get_sensitivity_single_trajectory(fnet,i,k,l,soln = None,shift = 0,self_inhibit=0,weights = None,mxTime = 1000):

    """
    Compute sensitivity of node i to parameter :math:`a_{kl}`. To do this, we need to solve an ODE that arises from the chain rule. Computes with 1 ODE solution, which may be sensitive to initial conditions (chosen at random)

    :param fnet: network of interactions for the model
    :type fnet: :py:obj:`friendlyNet <friendlyNet.friendlyNet>`
    :param i: index of node of interest
    :type i: int
    :param k: source node of interaction of interest
    :type k: int 
    :param l: target node of interaction of interest
    :type l: int
    :param soln: solution to lotka-volterra system *must use same shift and self_inhibit parameters*
    :type soln: scipy.integrate.solve_ivp object
    :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
    :type shift: float
    :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`a_{ii}`) will be set to -``self_inhibit``
    :type self_inhibit: float
    :param weights: weights for time-points of the ODE solution. This allows us to weight the later timepoints (closer to equilibrium) higher, or not, with some granularity. Default unweighted.
    :type weights: array[float]
    :param mxTime: time length of simulations
    :type mxTime: float

    :return: (possibly weighted) average value of :math:`\partial x_i/\partial a_{kl}` over time points in simulation.
    :rtype: float
    """

    
    if list(weights) == None:
        weights = []
    if soln == None:
        soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift = shift,self_inhibit = self_inhibit,dense_output = True)
    shtadj = fnet.Adjacency.T-shift
    np.fill_diagonal(shtadj,-self_inhibit)
    sensitivity = solve_ivp(sense_kl,(0,soln.t[-1]),np.zeros(fnet.Adjacency.shape[0]),args = (soln.sol,k,l,shtadj))
    if len(weights):
        weighted_avg = np.dot(sensitivity.y[i][-min(len(weights),len(sensitivity.y[i])):],weights[-min(len(weights),len(sensitivity.y[i])):])
        return weighted_avg#
    return np.mean(sensitivity.y[i])

def get_sensitivity(fnet,invadernode,target_node,source_node,soln = None,shift = 0,self_inhibit=0,mxTime = 1000,numtrials = 1000,wpts = 40,base_we = 1.5,nj=1):
    """
    Compute sensitivity of node i to parameter :math:`a_{kl}`. To do this, we need to solve an ODE that arises from the chain rule. Computes with 1000 ODE solutions with randomly seleced initial conditions

    :param fnet: network of interactions for the model
    :type fnet: :py:obj:`friendlyNet <friendlyNet.friendlyNet>`
    :param invadernode: index of node of interest
    :type invadernode: int/str
    :param source_node: source node of interaction of interest
    :type source_node: int/str 
    :param target_node: target node of interaction of interest
    :type target_node: int/str
    :param soln: solution to lotka-volterra system *must use same shift and self_inhibit parameters*
    :type soln: scipy.integrate.solve_ivp object
    :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
    :type shift: float
    :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`a_{ii}`) will be set to -``self_inhibit``
    :type self_inhibit: float
    :param wpts: number of time-points in each simulation to average over (will be final time-points). Default 40
    :type wpts: int
    :param base_we: base for weights of time-averaging - should be :math:`>1` for increasing weight, equal to 1 for uniform weight on last ``wpt`` time-points. Default 1.5
    :type base_we: float
    :param mxTime: time length of simulations
    :type mxTime: float
    :param numtrials: Number of simulations to average over
    :type numtrials: int
    :param nj: number of trials to run in parallel (using joblib)
    :type nj: int
    :return: (possibly weighted) average value of :math:`\partial x_i/\partial a_{kl}` over time points in simulation.
    :rtype: float
    """

    weights = np.array([base_we**ex for ex in range(wpts)])
    weights = weights/sum(weights)
    i = fnet.NodeNames.index(invadernode)
    k = fnet.NodeNames.index(source_node)
    l = fnet.NodeNames.index(target_node)
    mnsense = sum(Parallel(n_jobs = nj)(delayed(get_sensitivity_single_trajectory)(fnet,i,k,l,mxTime = mxTime,shift = shift,weights = weights,self_inhibit=self_inhibit) for tri in range(numtrials)))/numtrials

    return mnsense


def get_all_sensitivity_single_trajectory(fnet,i,shift = 0,self_inhibit=0,weights = None,mxTime = 1000,pars = 'all'):

    """
    Compute sensitivity of node i to every interaction parameter. To do this, we need to solve an ODE that arises from the chain rule. Because we can reuse the solution to the lotka-volerra
    system, we don't want to repeatedly call get_sensitivity on a each parameter.

    :param fnet: network of interactions for the model
    :type fnet: :py:obj:`friendlyNet <friendlyNet.friendlyNet>`
    :param i: index of node of interest
    :type i: int
    :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
    :type shift: float
    :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`a_{ii}`) will be set to -``self_inhibit``
    :type self_inhibit: float
    :param weights: weights for time-points of the ODE solution. This allows us to weight the later timepoints (closer to equilibrium) higher, or not, with some granularity. Default unweighted.
    :type weights: array[float]
    :param mxTime: time length of simulations
    :type mxTime: float
    :param pars: Which interaction parameters to test sensitivity to. If 'all', tests for every parameter. Otherwise, should be a list of tuples of indices (source,target). Default 'all'
    :type pars: list[tuple[int,int]]
    :return: (possibly weighted) average value of :math:`\partial x_i/\partial a_{kl}` over time points in simulation for each. If ``pars`` == 'all', returns NxN array indexed by [source,target]. Otherwise, returns 1d array corresponding to 'pars'
    :rtype: array[float]
    """

    if list(weights) == None:
        weights = []
    all_sensitivity = np.zeros_like(fnet.Adjacency)
    soln = fnet.solve_lotka_volterra(np.random.rand(fnet.Adjacency.shape[0]),mxTime,shift = shift,self_inhibit = self_inhibit,dense_output = True)
    try:
        if pars.lower() == 'all':
            srces = range(all_sensitivity.shape[0])
            trgts = range(all_sensitivity.shape[1])
        else:
            srces = [p[0] for p in pars]
            trgts = [p[1] for p in pars]
    except:
        srces = [p[0] for p in pars]
        trgts = [p[1] for p in pars]
    for k in srces:
        for l in trgts:
            all_sensitivity[k,l] = get_sensitivity_single_trajectory(fnet,i,k,l,soln = soln,shift = shift,self_inhibit = self_inhibit,weights = weights,mxTime = mxTime)
            # sensitivity = solve_ivp(sense_kl,(0,soln.t[-1]),np.zeros(fnet.Adjacency.shape[0]),args = (soln.sol,k,l,fnet.Adjacency.T-shift))
            # if len(weights):
            #     weighted_avg = np.dot(sensitivity.y[i][-min(len(weights),len(sensitivity.y[i])):],weights[-min(len(weights),len(sensitivity.y[i])):])
            # else:
            #     weighted_avg = np.mean(sensitivity.y[i])
            # all_sensitivity[k,l] = weighted_avg
            # print("{}      {}          {}/{}            ".format(sensitivity.t[-1],weighted_avg,l+k*all_sensitivity.shape[0],all_sensitivity.size),end = '\r')
    if pars == 'All':
        return all_sensitivity
    else:
        return np.array([all_sensitivity[p] for p  in pars])

def get_all_sensitivity(target_node,fnet,entries = 'all',shift = 0,self_inhibit=0,numtrials = 100,nj=1,mxTime = 1000,wpts = 40,base_we = 1.5):

    """
    Compute sensitivity of a node to interaction parameters, and average over simulations. Computes weighted average, using *final* ``wpts`` timepoints with increasing weight. Weight s is computed as :math:`b^s` where :math:`b` is ``base_we`` and then weights are rescaled to sum to 1. 

    :param target_node: name node of interest (nodes are usually named with str, but can be named with other objects, most commonly int equal to node index.)
    :type target_node: str or int
    :param fnet: network of interactions for the model
    :type fnet: :py:obj:`friendlyNet <friendlyNet.friendlyNet>`
    :param entries: Which interaction parameters to test sensitivity to (names). If 'all', tests for every parameter. Otherwise, should be a list of tuples of names (source,target). Default 'all'
    :type entries: list[tuple[str,str]]
    :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
    :type shift: float
    :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`a_{ii}`) will be set to -``self_inhibit``
    :type self_inhibit: float
    :param numtrials: Number of simulations to average over
    :type numtrials: int
    :param nj: number of trials to run in parallel (using joblib)
    :type nj: int
    :param mxTime: time length of simulations
    :type mxTime: float
    :param wpts: number of time-points in each simulation to average over (will be final time-points). Default 40
    :type wpts: int
    :param base_we: base for weights of time-averaging - should be :math:`>1` for increasing weight, equal to 1 for uniform weight on last ``wpt`` time-points. Default 1.5
    :type base_we: float

    :return: Average value of :math:`\partial x_i/\partial a_{kl}` averaged first over time points in each simulation and next over simulations. If ``pars`` == 'all', returns NxN array indexed by [source,target]. Otherwise, returns 1d array corresponding to 'pars'
    :rtype: float
    """


    weights = np.array([base_we**ex for ex in range(wpts)])
    weights = weights/sum(weights)
    i = fnet.NodeNames.index(target_node)
    pars = [(fnet.NodeNames.index(ent[0]),fnet.NodeNames.index(ent[1])) for ent in entries]
    mnsense = sum(Parallel(n_jobs = nj)(delayed(get_all_sensitivity_single_trajectory)(fnet,i,pars=pars,mxTime = mxTime,shift = shift,weights = weights,self_inhibit=self_inhibit) for tri in range(numtrials)))/numtrials
    try:
        if entries == 'all':
            sensitivities = pd.DataFrame(mnsense.T, columns = fnet.NodeNames,index = fnet.NodeNames)
        else:
            sensitivities = dict([(entries[i],mnsense[i]) for i in range(len(entries))])
    except:
        sensitivities = dict([(entries[i],mnsense[i]) for i in range(len(entries))])


    return sensitivities



