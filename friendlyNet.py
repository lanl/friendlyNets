import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel,delayed

class friendlyNet:

    """
        Network class designed for testing how positive a directed network is for a node. Provides several ways to score the network's friendliness for a given node. By friendliness,
        we mean some measure of how much the network promotes/inhibits a node based on a choice of interaction model represented by the graph (e.g. Lotka-Volterra or Diffusion).
        Rescales the adjacency matrix so that all weights are in :math:`[0,1]`

        :param adj: Adjacency matrix of the graph.
        :type adj: (N,N) array[float]

    """    

    def __init__(self,adj):

        self.Adjacency = np.array(adj)/abs(np.array(adj)).max()
        """Network adjacency matrix, rescaled so that all weights are in :math:`[0,1]`"""
        self.NodeNames = []
        """List of names of the nodes, corresponding to ordering of adjacency matrix"""
        self.InDegree = np.diag(self.Adjacency.sum(axis = 1))
        """Total weight of edges into each node"""
        self.OutDegree = np.diag(self.Adjacency.sum(axis = 0))
        """Total weight of edges out of each node"""
        self.NodeScores = None
        """Pandas DataFrame of friendliness scores for each node for a set of chosen dynamical models."""
        self.EdgeList = None
        """Pandas DataFrame of edges (with weights) for the network that can be easily saved and loaded into cytoscape. See :py:func:`make_edge_list <friendlyNet.friendlyNet.make_edge_list>`"""

    def make_edge_list(self):

        """
        Create a list of edges with columns:

        - Source
        - Target
        - Weight
        - ABS_Weight (absolute value)
        - Sign_Weight

        **Modifies**

        - :py:obj:`EdgeList <friendlyNet.friendlyNet.EdgeList>`

        :return: None

        """

        if len(self.NodeNames) != self.Adjacency.shape[0]:
            self.NodeNames = list(range(self.Adjacency.shape[0]))

        num_edges = np.count_nonzero(self.Adjacency)
        edges = pd.DataFrame(index = range(num_edges), columns = ["Source","Target","Weight","ABS_Weight","Sign_Weight"])
        indx = 0
        for i in range(self.Adjacency.shape[0]):
            for j in range(self.Adjacency.shape[1]):
                if self.Adjacency[i,j] != 0:
                    edges.loc[indx] = [self.NodeNames[j],self.NodeNames[i],self.Adjacency[i,j],abs(self.Adjacency[i,j]),np.sign(self.Adjacency[i,j])]
                    indx+=1

        self.EdgeList = edges

    def lotka_volterra_system(self,t,s,shift,self_inhibit):

        """
        Right-Hand side of generalized Lotka-Volterra dynamical system, using :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` for interactions:

        .. math::
        
            \dot{x}_i = x_i(1+\sum_{j=1}^N a_{ij} x_j)

        :param t: time in simulation
        :type t: float
        :param s: State of simulation (species abundance)
        :type s: array[float]
        :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
        :type shift: float
        :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`a_{ii}`) will be set to -``self_inhibit``
        :type self_inhibit: float

        :return: value of vector field at t,s
        :rtype: array[float]
        """

        adjc = self.Adjacency.copy()
        np.fill_diagonal(adjc,-self_inhibit)
        return s*(1+np.dot(adjc - shift,s))

    def solve_lotka_volterra(self,s0,T,shift=0,bup = 1000,self_inhibit = 0):

        """
        Solves the generalized Lotka-Volterra dynamical system, using :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` for interactions

        :param s0: Inital state of simulation (species abundance)
        :type s0: array[float]
        :param T: Simulation length
        :type T: float
        :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
        :type shift: float
        :param bup: maximum abundance to allow in simulation. If any state variable reaches this value, it is assumed that the simulation has exhibited finite-time blowup and the simulation is stopped.
        :type bup: float
        :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`A_{ii}`) will be set to negative ``self_inhibit``
        :type self_inhibit: float

        :return: Solution to the Lotka-Volterra dynamics 
        :rtype: scipy.integrate.solve_ivp solution
        """


        blowup = lambda t,s:sum(s) - bup
        blowup.terminal = True
        thefun = lambda t,s: self.lotka_volterra_system(t,s,shift,self_inhibit)
        sln = solve_ivp(thefun,(0,T),s0,dense_output = True,events = blowup)
        return sln

    # def sample_lotka_volterra_unif(self,g,node,dist_params,shift=0,self_inhibit=0):

    #     """
    #     Sample :math:`f_i(s)` for node i 
    #     """
    #     s = dist_params[0] + (dist_params[1]-dist_params[0])*g.random(self.Adjacency.shape[0])
    #     dsdt = self.lotka_volterra_system(0,s,shift,self_inhibit)
    #     return dsdt[node]

    # def sample_lotka_volterra_lognorm(self,g,node,dist_params,shift=0,self_inhibit=0):

    #     """
    #     Sample :math:`f_i(s)` for node i 
    #     """
    #     logs = g.multivariate_normal(dist_params[0],dist_params[1])
    #     s = np.exp(logs)
    #     dsdt = self.lotka_volterra_system(0,s,shift,self_inhibit)
    #     return dsdt[node]

    # def sample_lotka_volterra_poisson(self,g,node,dist_params,shift=0,self_inhibit=0):

    #     """
    #     Sample :math:`f_i(s)` for node i 
    #     """
    #     s = g.poisson(lam = dist_params,size = self.Adjacency.shape[0])
    #     dsdt = self.lotka_volterra_system(0,s,shift,self_inhibit)
    #     return dsdt[node]

    # def lotka_volterra_score(self,node,numSamples,shift=0,self_inhibit=0,distr = "uniform", dist_params = None,nj=1):

    #     """
    #     Sample 1/|s|f_i(s) from the LV system for node i

    #     :param node: name or index of node
    #     :type node: str or int
    #     :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
    #     :type shift: float
    #     :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`A_{ii}`) will be set to negative ``self_inhibit``
    #     :type self_inhibit: float
    #     :param distr:
    #     :type distr: str
    #     :param dist_params:
    #     type dist_params: list[float]

    #     :return: expecation of :math:`\\frac{1}{\|s\|}f_i(s)` from given distribution
    #     :rtype:  float
    #     """

    #     g = np.random.default_rng()

    #     if distr.lower() == "uniform":
    #         if dist_params == None:
    #             dist_params = (0,1)
    #         samples = Parallel(n_jobs = nj)(delayed(self.sample_lotka_volterra_unif)(g,node,dist_params,shift=shift,self_inhibit=self_inhibit) for i in range(numSamples))

    #     elif distr.lower() == "lognormal":
    #         if dist_params == None:
    #             dist_params = (np.zeros(self.Adjacency.shape[0]),np.eye(self.Adjacency.shape[0]))
    #         samples = Parallel(n_jobs = nj)(delayed(self.sample_lotka_volterra_lognorm)(g,node,dist_params,shift=shift,self_inhibit=self_inhibit) for i in range(numSamples))
        
    #     elif distr.lower() == "poisson":
    #         if dist_params == None:
    #             dist_params = 1
    #         samples = Parallel(n_jobs = nj)(delayed(self.sample_lotka_volterra_poisson)(g,node,dist_params,shift=shift,self_inhibit=self_inhibit) for i in range(numSamples))


    #     return np.mean(samples)



    def lotka_volterra_score_single(self,node,mxTime = 100,shift=0,self_inhibit = 0):

        """
        Uses the Lotka-Volterra dynamical system to determine the network's friendliness to a particular node. Solves the system using :py:func:`solve_lotka_volterra <friendlyNet.friendlyNet.solve_lotka_volterra>`.
        Solves the Lotka-Volterra system with random initial conditions and computes a score. The score is based on final relative abundance, but for finer scoring we also account for time to extinction
        and time to domination (e.g. relative abundance near 1). The score is computed as 

        .. math
            (T_e + T_d + r)/3
        
        where :math:`T_e` is the proportion of the time internal that the species is *not* extinct, :math:`T_d` is the proportion of the time interval that the species *is* dominant, and :math:`r` is the final relative abundance of the species.

        :param node: name or index of node
        :type node: str or int
        :param mxTime: time length of simulations
        :type mxTime: float
        :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
        :type shift: float
        :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`A_{ii}`) will be set to negative ``self_inhibit``
        :type self_inhibit: float

        :return: Friendliness of the network to the node, according to the single Lotka-Volterra simulation, and the status of the ODE solution
        :rtype:  tuple[float,str]
        """

        if node in self.NodeNames:
            node = list(self.NodeNames).index(node)

        long_time = self.solve_lotka_volterra(np.random.rand(self.Adjacency.shape[0]),mxTime,shift=shift,self_inhibit = self_inhibit)

        all_relative = long_time.y/(long_time.y.sum(axis = 0))
        final_relative = all_relative[node,-1]

        time_below = np.where(all_relative[node] < 10**-5)
        time_above = np.where(all_relative[node] > 1-10**-5)

        if len(time_below[0]):
            extinction = long_time.t[time_below[0][0]]/mxTime
        else:
            extinction = 1
        if len(time_above[0]):
            domination = 1 - long_time.t[time_above[0][0]]/mxTime
        else:
            domination = 0

        score = extinction + domination + final_relative.round(7)

        if long_time.status != -1:
            return score/3,long_time.status

    def lotka_volterra_score(self,node,mxTime = 100,numtrials = 1000,nj = -1,shift=0,self_inhibit = 0,cntbu = False):

        """

        Provides a score using repeated trials of the Lotka-Volterra system. Scores using :py:func:`lotka_volterra_score_single <friendlyNet.friendlyNet.lotka_volterra_score_single>`

        :param node: name or index of node
        :type node: str or int
        :param mxTime: time length of simulations
        :type mxTime: float
        :param numtrials: Number of ODE solutions and corresponding scores to compute
        :type numtrials: int
        :param nj: Number of parrallel simulations to run concurrently (uses joblib)
        :type nj: int
        :param shift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``
        :type shift: float
        :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`A_{ii}`) will be set to negative ``self_inhibit``
        :type self_inhibit: float
        :param cntbu: Whether or not to count the number of blow-ups in the simulations
        :type cntbu: bool

        :return: score from :py:func:`lotka_volterra_score_single <friendlyNet.friendlyNet.lotka_volterra_score_single>` averaged over all trials, optinoally number of blowups in simulation
        :rtype: float,int
        """


        trials = Parallel(n_jobs = nj)(delayed(self.lotka_volterra_score_single)(node,mxTime = mxTime,shift=shift,self_inhibit = self_inhibit) for i in range(numtrials))
        if cntbu:
            return (np.mean([val[0] for val in trials if val != None]),np.mean([float(val[1]) for val in trials if val != None]))
        else:
            return np.mean([val[0] for val in trials if val != None])

    def replicator_system(self,t,s):

        """
        Right-Hand side of replicator dynamical system, using :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` for interactions:

        .. math::
        
            \dot{x}_i = x_i(\sum_{j=1}^N a_{ij} x_j - x^TAx)

        :param t: time in simulation
        :type t: float
        :param s: State of simulation (species abundance)
        :type s: array[float]

        :return: value of vector field at t,s
        :rtype: array[float]
        """


        return s*(np.dot(self.Adjacency,s) - np.dot(s.T,np.dot(self.Adjacency,s.T)))

    def solve_replicator(self,s0,T):

        """
        Solves the replicator dynamical system, using :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` for interactions

        :param s0: Inital state of simulation (species abundance)
        :type s0: array[float]
        :param T: Simulation length
        :type T: float

        :return: Solution to the replicator dynamics 
        :rtype: scipy.integrate.solve_ivp solution
        """


        blowup = lambda t,s:sum(s) - 1000
        blowup.terminal = True
        sln = solve_ivp(self.replicator_system,(0,T),s0,dense_output = True,events = blowup)
        return sln

    def replicator_score_single(self,node,mxTime = 100):


        """
        Uses the replicator dynamical system to determine the network's friendliness to a particular node. Solves the system using :py:func:`solve_replicator <friendlyNet.friendlyNet.solve_replicator>`.
        Solves the replicator system with random initial conditions and computes a score. The score is based on final relative abundance, but for finer scoring we also account for time to extinction
        and time to domination (e.g. relative abundance near 1). The score is computed as 

        .. math
            (T_e + T_d + r)/3
        
        where :math:`T_e` is the proportion of the time internal that the species is *not* extinct, :math:`T_d` is the proportion of the time interval that the species *is* dominant, and :math:`r` is the final relative abundance of the species.

        :param node: name or index of node
        :type node: str or int
        :param mxTime: time length of simulations
        :type mxTime: float

        :return: Friendliness of the network to the node, according to the single replicator simulation, and the status of the ODE solution
        :rtype:  tuple[float,str]
        """


        if node in self.NodeNames:
            node = list(self.NodeNames).index(node)

        long_time = self.solve_replicator(np.random.rand(self.Adjacency.shape[0]),mxTime)

        all_relative = long_time.y/(long_time.y.sum(axis = 0))
        final_relative = all_relative[node,-1]

        time_below = np.where(all_relative[node] < 10**-5)
        time_above = np.where(all_relative[node] > 1-10**-5)

        if len(time_below[0]):
            extinction = long_time.t[time_below[0][0]]/mxTime
        else:
            extinction = 1
        if len(time_above[0]):
            domination = 1 - long_time.t[time_above[0][0]]/mxTime
        else:
            domination = 0

        score = extinction + domination + final_relative.round(7)

        return score/3

    def replicator_score(self,node,mxTime = 100,numtrials = 1000,nj=-1):

        """

        Provides a score using repeated trials of the replicator system. Scores using :py:func:`replicator_score_single <friendlyNet.friendlyNet.replicator_score_single>`

        :param node: name or index of node
        :type node: str or int
        :param mxTime: time length of simulations
        :type mxTime: float
        :param numtrials: Number of ODE solutions and corresponding scores to compute
        :type numtrials: int
        :param nj: Number of parrallel simulations to run concurrently (uses joblib)
        :type nj: int


        :return: score from :py:func:`lotka_volterra_score_single <friendlyNet.friendlyNet.lotka_volterra_score_single>` averaged over all trials
        :rtype: float
        """

        return np.mean(Parallel(n_jobs = nj)(delayed(self.replicator_score_single)(node,mxTime = mxTime) for i in range(numtrials)))

    def node_balanced_score(self,node):

        """
        Provides a score based on the linear system 

        .. math:
            \dot{x} = L x
        
        where :math:`L= A^T-D` is the graph laplance matrix for the graph after weights have been rescaled to the interval :math:`[0,1]`.
        This dynamical system arises from the notion of node-balancing the graph. Because this is linear, we can use the dominant eigenvector of the
        laplacian to compute equilibrium.

        :param node: name or index of node
        :type node: str or int
        
        :return: Value of node in dominant eigenvector (i.e. equilibrium solution)
        :rtype: float
        """

        if node in self.NodeNames:
            node = list(self.NodeNames).index(node)

        rescld_Adj = (self.Adjacency + 1)/2
        L = rescld_Adj - np.diag(rescld_Adj.sum(axis = 0))
        eigenvalues,eigenvectors = np.linalg.eig(L)

        dominant_eval = np.where(np.real(eigenvalues) == np.real(eigenvalues).max())
        dominant_evec = eigenvectors[:,dominant_eval[0]].sum(axis = 1)
        dominant_evec = dominant_evec/dominant_evec.sum()

        return np.real(dominant_evec)[node]

    def node_balanced_system(self,T):

        """
        Function to provide a simulation for the node-balancing linear system 

        .. math:
            \dot{x} = L x
        
        where :math:`L= A^T-D` is the graph laplance matrix for the graph after weights have been rescaled to the interval :math:`[0,1]`.

        :param T: End time of simulation
        :type T: float
        
        :return: Solution to the node balance dynamics 
        :rtype: scipy.integrate.solve_ivp solution
        """

        rescld_Adj = (self.Adjacency + 1)/2
        L = rescld_Adj - np.diag(rescld_Adj.sum(axis = 0))

        return solve_ivp(lambda t,s: np.dot(L,s), (0,T), [1,1,1])

    def stochastic_score(self,node):

        """
        Provides a score based on the linear system that simulates concurrent random walks (or, equivalently, diffusion) on the graph.
        We rescale the adjacency matrix to build a stochastic matrix that represents the transition probabilities in a random walk on the graph.
        The eigenvectors of this matrix provide a stationary distribution for the cuncurrent random walks.

        :param node: name or index of node
        :type node: str or int
        
        :return: Value of node in stationary distribution
        :rtype: float
        """

        if node in self.NodeNames:
            node = list(self.NodeNames).index(node)

        rescld_Adj = (self.Adjacency + 1)/2
        stoch_matrix = (rescld_Adj.T/rescld_Adj.sum(axis = 1))

        row_eigenvals,row_eigenvecs = np.linalg.eig(stoch_matrix)
        stationary_indx = np.where(row_eigenvals.round(7) == 1)

        stationary_distribution = row_eigenvecs[:,stationary_indx[0]].sum(axis = 1)
        stationary_distribution = stationary_distribution/stationary_distribution.sum()

        return np.real(stationary_distribution)[node]

    def score_node(self,node,scores = None,odeTrials = None):

        """
        Function to score a node using a set of scores. Choose any list of the following:

        - **LV** The :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>`
        - **InhibitLV** :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>` with self inhibition = 1
        - **AntLV** The :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>` with all interactions shifted by -1 to make them antagonistic.
        - **Replicator** The :py:func:`replicator equation dynamics <friendlyNet.friendlyNet.replicator_system>` 
        - **NodeBalance** The linear :py:func:`node balancing dynamical system <friendlyNet.friendlyNet.node_balanced_score>` 
        - **Stochastic** The linear :py:func:`random walk dynamical system <friendlyNet.friendlyNet.stochastic_score>` 

        A composite score will be included, which is simply the mean of each score included.

        :param node: name or index of node
        :type node: int or str
        :param scores: list of score types you wish to use. Leave as None for all 6.
        :type scores: list
        :param odeTrials: Number of simulations of the Lotka-Volterra and replicator dynamics to use to estimate the score. Leave as None for number of trials equal to number of nodes in the network.
        :type odeTrials: int

        :return: Dictionary of scores keyed by score type.
        :trype: dict[str,float]
        

        """

        if scores == None:
            scores = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic","Composite"]


        all_scores = {}
        if isinstance(odeTrials,int):
            if "LV" in scores:
                all_scores["LV"] = self.lotka_volterra_score(node,numtrials = odeTrials)
            if "InhibitLV" in scores:
                all_scores["InhibitLV"] = self.lotka_volterra_score(node,numtrials = odeTrials,self_inhibit=1)
            if "AntLV" in scores:
                all_scores["AntLV"] = self.lotka_volterra_score(node,numtrials = odeTrials,shift = 1)
            if "Replicator" in scores:
                all_scores["Replicator"] = self.replicator_score(node,numtrials = odeTrials)
        else:
            if "LV" in scores:
                all_scores["LV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0])
            if "InhibitLV" in scores:
                all_scores["InhibitLV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0],self_inhibit=1)
            if "AntLV" in scores:
                all_scores["AntLV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0],shift=1)
            if "Replicator" in scores:
                all_scores["Replicator"] = self.replicator_score(node,numtrials = self.Adjacency.shape[0])
        if "NodeBalance" in scores:
            all_scores["NodeBalance"] = self.node_balanced_score(node)
        if "Stochastic" in scores:
            all_scores["Stochastic"] = self.stochastic_score(node)
        all_scores["Composite"] = np.mean(list(all_scores.values()))
        
        return all_scores

    def score_all_nodes(self,scores = None,odeTrials = None):

        """
        Function to score all nodes using a set of scores. Choose any list of the following:

        - **LV** The :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>`
        - **InhibitLV** :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>` with self inhibition = 1
        - **AntLV** The :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>` with all interactions shifted by -1 to make them antagonistic.
        - **Replicator** The :py:func:`replicator equation dynamics <friendlyNet.friendlyNet.replicator_system>` 
        - **NodeBalance** The linear :py:func:`node balancing dynamical system <friendlyNet.friendlyNet.node_balanced_score>` 
        - **Stochastic** The linear :py:func:`random walk dynamical system <friendlyNet.friendlyNet.stochastic_score>` 

        A composite score will be included, which is simply the mean of each score included.

        :param scores: list of score types you wish to use. Leave as None for all 6.
        :type scores: list
        :param odeTrials: Number of simulations of the Lotka-Volterra and replicator dynamics to use to estimate the score. Leave as None for number of trials equal to number of nodes in the network.
        :type odeTrials: int

        :return: Table of scores for each node and score type chosen
        :trype: pandas dataframe
        
        **Modifies**

        - :py:obj:`NodeScores <friendlyNet.friendlyNet.NodeScores>`

        """

        if scores == None:
            scores = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic"]
            
        scores = scores + ["Composite"]


        if len(self.NodeNames) != self.Adjacency.shape[0]:
            self.NodeNames = list(range(self.Adjacency.shape[0]))

        scores = pd.DataFrame(index = self.NodeNames,columns = scores)
        for nd in self.NodeNames:
            scrs = self.score_node(nd,scores = scores, odeTrials=odeTrials)
            scores.loc[nd] = [scrs[col] for col in scores.columns]

        self.NodeScores = scores

        return scores
