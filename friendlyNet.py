import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from joblib import Parallel,delayed

class friendlyNet:

    def __init__(self,adj):

        self.Adjacency = np.array(adj)/abs(np.array(adj)).max()
        self.NodeNames = []
        self.InDegree = np.diag(self.Adjacency.sum(axis = 1))
        self.OutDegree = np.diag(self.Adjacency.sum(axis = 0))
        self.NodeScores = None
        self.EdgeList = None

    def make_edge_list(self):

        if len(self.NodeNames) != self.Adjacency.shape[0]:
            self.NodeNames = list(range(self.Adjacency.shape[0]))

        num_edges = np.count_nonzero(self.Adjacency)
        edges = pd.DataFrame(index = range(num_edges), columns = ["Source","Target","Interaction","ABS_Interaction","Distance"])
        indx = 0
        for i in range(self.Adjacency.shape[0]):
            for j in range(self.Adjacency.shape[1]):
                if self.Adjacency[i,j] != 0:
                    edges.loc[indx] = [self.NodeNames[i],self.NodeNames[j],self.Adjacency[i,j],abs(self.Adjacency[i,j]),1/abs(self.Adjacency[i,j])]
                    indx+=1

        self.EdgeList = edges

    def lotka_volterra_system(self,t,s,shift,self_inhibit):
        adjc = self.Adjacency.copy().T
        np.fill_diagonal(adjc,-self_inhibit)
        return s*(1+np.dot(adjc - shift,s))

    def solve_lotka_volterra(self,s0,T,shift=0,bup = 1000,self_inhibit = 0):
        blowup = lambda t,s:sum(s) - bup
        blowup.terminal = True
        thefun = lambda t,s: self.lotka_volterra_system(t,s,shift,self_inhibit)
        sln = solve_ivp(thefun,(0,T),s0,dense_output = True,events = blowup)
        return sln

    def lotka_volterra_score_single(self,node,mxTime = 100,shift=0,self_inhibit = 0):

        if node in self.NodeNames:
            node = self.NodeNames.index(node)

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
        trials = Parallel(n_jobs = nj)(delayed(self.lotka_volterra_score_single)(node,mxTime = mxTime,shift=shift,self_inhibit = self_inhibit) for i in range(numtrials))
        if cntbu:
            return (np.mean([val[0] for val in trials if val != None]),np.mean([float(val[1]) for val in trials if val != None]))
        else:
            return np.mean([val[0] for val in trials if val != None])

    def antagonistic_lotka_volterra_system(self,t,s):
        return s*(1+np.dot(self.Adjacency.T-1,s))

    def solve_antagonistic_lotka_volterra(self,s0,T):
        blowup = lambda t,s:sum(s) - 1000
        blowup.terminal = True
        sln = solve_ivp(self.antagonistic_lotka_volterra_system,(0,T),s0,dense_output = True,events = blowup)
        return sln

    def antagonistic_lotka_volterra_score_single(self,node,mxTime = 100):

        if node in self.NodeNames:
            node = self.NodeNames.index(node)

        long_time = self.solve_antagonistic_lotka_volterra(np.random.rand(self.Adjacency.shape[0]),mxTime)

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

    def antagonistic_lotka_volterra_score(self,node,mxTime = 100,numtrials = 1000,nj=-1):
        return np.mean(Parallel(n_jobs = nj)(delayed(self.antagonistic_lotka_volterra_score_single)(node,mxTime = mxTime) for i in range(numtrials)))

    def replicator_system(self,t,s):
        return s*(np.dot(self.Adjacency.T,s) - np.dot(s.T,np.dot(self.Adjacency.T,s.T)))

    def solve_replicator(self,s0,T):
        blowup = lambda t,s:sum(s) - 1000
        blowup.terminal = True
        sln = solve_ivp(self.replicator_system,(0,T),s0,dense_output = True,events = blowup)
        return sln

    def replicator_score_single(self,node,mxTime = 100):

        if node in self.NodeNames:
            node = self.NodeNames.index(node)

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
        return np.mean(Parallel(n_jobs = nj)(delayed(self.replicator_score_single)(node,mxTime = mxTime) for i in range(numtrials)))

    def node_balanced_score(self,node):

        if node in self.NodeNames:
            node = self.NodeNames.index(node)

        rescld_Adj = (self.Adjacency.T + 1)/2
        L = rescld_Adj - np.diag(rescld_Adj.sum(axis = 0))
        eigenvalues,eigenvectors = np.linalg.eig(L)

        dominant_eval = np.where(np.real(eigenvalues) == np.real(eigenvalues).max())
        dominant_evec = eigenvectors[:,dominant_eval[0]].sum(axis = 1)
        dominant_evec = dominant_evec/dominant_evec.sum()

        return np.real(dominant_evec)[node]

    def node_balanced_system(self,T):

        rescld_Adj = (self.Adjacency.T + 1)/2
        L = rescld_Adj - np.diag(rescld_Adj.sum(axis = 0))

        return solve_ivp(lambda t,s: np.dot(L,s), (0,T), [1,1,1])

    def stochastic_score(self,node):
        if node in self.NodeNames:
            node = self.NodeNames.index(node)

        rescld_Adj = (self.Adjacency + 1)/2
        stoch_matrix = (rescld_Adj.T/rescld_Adj.sum(axis = 1))

        row_eigenvals,row_eigenvecs = np.linalg.eig(stoch_matrix)
        stationary_indx = np.where(row_eigenvals.round(7) == 1)

        stationary_distribution = row_eigenvecs[:,stationary_indx[0]].sum(axis = 1)
        stationary_distribution = stationary_distribution/stationary_distribution.sum()

        return np.real(stationary_distribution)[node]

    def score_node(self,node,odeTrials = None):
        all_scores = {}
        if isinstance(odeTrials,int):
            all_scores["LV"] = self.lotka_volterra_score(node,numtrials = odeTrials)
            all_scores["InhibitLV"] = self.lotka_volterra_score(node,numtrials = odeTrials,self_inhibit=1)
            all_scores["AntLV"] = self.lotka_volterra_score(node,numtrials = odeTrials,shift = 1)#self.antagonistic_lotka_volterra_score(node,numtrials = odeTrials)
            all_scores["Replicator"] = self.replicator_score(node,numtrials = odeTrials)
        else:
            all_scores["LV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0])
            all_scores["InhibitLV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0],self_inhibit=1)
            all_scores["AntLV"] = self.lotka_volterra_score(node,numtrials = self.Adjacency.shape[0],shift=1)#self.antagonistic_lotka_volterra_score(node,numtrials = self.Adjacency.shape[0])
            all_scores["Replicator"] = self.replicator_score(node,numtrials = self.Adjacency.shape[0])
        all_scores["NodeBalance"] = self.node_balanced_score(node)
        all_scores["Stochastic"] = self.stochastic_score(node)
        all_scores["Composite"] = np.mean(list(all_scores.values()))
        return all_scores

    def score_all_nodes(self):

        if len(self.NodeNames) != self.Adjacency.shape[0]:
            self.NodeNames = list(range(self.Adjacency.shape[0]))

        scores = pd.DataFrame(index = self.NodeNames,columns = ["LV","AntLV","InhibitLV","Replicator","NodeBalance","Stochastic","Composite"])
        for nd in self.NodeNames:
            scrs = self.score_node(nd)
            scores.loc[nd] = [scrs[col] for col in scores.columns]

        self.NodeScores = scores
