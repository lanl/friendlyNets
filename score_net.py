#### Score networks based on friendliness vs given score for each method of friendliness (and all)

### Input:
###   Overall Network
###   Relative abundance of each sample (for constructing induced subgraphs)
###   Name of node of interest (must be in Overall Network, need not be in sample - will be added to induced subgraph)
###   Score of each sample (binary or real number in [0,1] based on known classification or performance)

### Output:
###   If given real number score, Spearman rank correlation of score with friendliness
###   If given binary classification, return ROC curve and AUROC
from friendlyNet import *
from sklearn.metrics import roc_auc_score,roc_curve
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr,kendalltau

def score_net(experiment,full_net,target_node,scoretype, min_ra = 10**-6, odeTrials = None):

    net_scores = pd.DataFrame(index = experiment.keys(),columns = ["Score","LV","AntLV","InhibitLV","Replicator","NodeBalance","Stochastic","Composite"])

    for ky,sample in experiment.items():
        score = sample[0]
        data = sample[1]
        nonzero = [ky for ky,val in data.items() if val > min_ra]
        if target_node not in nonzero:
            nonzero += [target_node]

        subgraph = full_net.loc[nonzero,nonzero]
        friendly = friendlyNet(subgraph.values)
        friendly.NodeNames = nonzero

        fscores = friendly.score_node(target_node,odeTrials = odeTrials)

        net_scores.loc[ky] = [score] + [fscores[col] for col in net_scores.columns[1:]]

    if scoretype == 'b':

        aurocs = {}
        roc_curves = {}
        for col in net_scores.columns[1:]:
            aurocs[col] = roc_auc_score(net_scores["Score"].values.astype(float),net_scores[col].values.astype(float))
            roc_curves[col] = roc_curve(net_scores["Score"].values.astype(float),net_scores[col].values.astype(float))

        return net_scores,aurocs,roc_curves,np.mean([val for val in aurocs.values()])

    else:

        pearsonval_r = {}
        kendallval_r = {}
        spearmanval_r = {}
        pearsonp = {}
        kendallp = {}
        spearmanp = {}
        for col in net_scores.columns[1:]:
            pearsonval_r[col],pearsonp[col] = pearsonr(net_scores["Score"].values.astype(float),net_scores[col].values.astype(float))
            kendallval_r[col],kendallp[col] = kendalltau(net_scores["Score"].values.astype(float),net_scores[col].values.astype(float))
            spearmanval_r[col],spearmanp[col] = spearmanr(net_scores["Score"].values.astype(float),net_scores[col].values.astype(float))

        pearsonval = dict([(ky,(val+1)/2) for ky,val in pearsonval_r.items()])
        kendallval = dict([(ky,(val+1)/2) for ky,val in kendallval_r.items()])
        spearmanval = dict([(ky,(val+1)/2) for ky,val in spearmanval_r.items()])


        return net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp

def score_light(experiment,full_net,target_node,scoretype, score_model,self_inhibit = 0, min_ra = 10**-6, odeTrials = None,lvshift = 0,cntbu = False,keepscores = False,KO = None):

    net_scores = np.empty(len(experiment),dtype = np.float64)
    net_test_scores = np.empty(len(experiment),dtype = np.float64)
    indx = 0
    sample_order = list(experiment.keys())

    if cntbu:
        blowupcounts = np.empty(len(experiment),dtype = np.float64)

    for key in sample_order:
        sample = experiment[key]
        net_scores[indx] = sample[0]
        data = sample[1]
        nonzero = [ky for ky,val in data.items() if val > min_ra]
        if target_node not in nonzero:
            nonzero += [target_node]
        
        if KO in nonzero:
            nonzero.remove(KO)

        subgraph = full_net.loc[nonzero,nonzero]
        friendly = friendlyNet(subgraph.values)
        friendly.NodeNames = nonzero

        if score_model == "LV":
            if cntbu:
                if isinstance(odeTrials,int):
                    r = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = lvshift,cntbu=cntbu,self_inhibit=self_inhibit)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
                else:
                    r = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = lvshift,cntbu=cntbu,self_inhibit=self_inhibit)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
            else:
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = lvshift,self_inhibit=self_inhibit)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = lvshift,self_inhibit=self_inhibit)
        elif score_model == "AntLV":
            if cntbu:
                if isinstance(odeTrials,int):
                    r = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 1,cntbu=cntbu,self_inhibit=0)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
                else:
                    r = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 1,cntbu=cntbu,self_inhibit=0)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
            else:
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 1,self_inhibit=0)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 1,self_inhibit=0)
        elif score_model == "InhibitLV":
            if cntbu:
                if isinstance(odeTrials,int):
                    r = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 0,cntbu=cntbu,self_inhibit=1)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
                else:
                    r = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 0,cntbu=cntbu,self_inhibit=1)
                    net_test_scores[indx] = r[0]
                    blowupcounts[indx] = r[1]
            else:
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 0,self_inhibit=1)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 0,self_inhibit=1)                
        elif score_model == "Replicator":
            if isinstance(odeTrials,int):
                net_test_scores[indx] = friendly.replicator_score(target_node,numtrials = odeTrials)
            else:
                net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
        elif score_model == "NodeBalance":
            net_test_scores[indx] = friendly.node_balanced_score(target_node)
        elif score_model == "Stochastic":
            net_test_scores[indx] = friendly.stochastic_score(target_node)
        elif score_model == "Composite":
            net_test_scores[indx] = friendly.score_node(target_node,odeTrials = odeTrials)["Composite"]
        indx += 1

    if keepscores:
        if cntbu:
            if scoretype == 'b':

                return roc_auc_score(net_scores,net_test_scores),np.mean(blowupcounts),sample_order,net_test_scores

            else:

                kendallval,_ = kendalltau(net_scores,net_test_scores)
                spearmanval,_ = spearmanr(net_scores,net_test_scores)


                return kendallval,spearmanval,np.mean(blowupcounts),sample_order,net_test_scores
        else:
            if scoretype == 'b':

                return roc_auc_score(net_scores,net_test_scores),sample_order,net_test_scores

            else:

                kendallval,_ = kendalltau(net_scores,net_test_scores)
                spearmanval,_ = spearmanr(net_scores,net_test_scores)


                return kendallval,spearmanval,sample_order,net_test_scores
    else:
        if cntbu:
            if scoretype == 'b':

                return roc_auc_score(net_scores,net_test_scores),np.mean(blowupcounts)

            else:

                kendallval,_ = kendalltau(net_scores,net_test_scores)
                spearmanval,_ = spearmanr(net_scores,net_test_scores)


                return kendallval,spearmanval,np.mean(blowupcounts)
        else:
            if scoretype == 'b':

                return roc_auc_score(net_scores,net_test_scores)

            else:

                kendallval,_ = kendalltau(net_scores,net_test_scores)
                spearmanval,_ = spearmanr(net_scores,net_test_scores)


                return kendallval,spearmanval

def score_binary(experiment,full_net,target_node, score_model, min_ra = 10**-6, odeTrials = None):

        net_scores = np.empty(len(experiment),dtype = np.float64)
        net_test_scores = np.empty(len(experiment),dtype = np.float64)
        indx = 0

        for ky,sample in experiment.items():
            net_scores[indx] = sample[0]
            data = sample[1]
            nonzero = [ky for ky,val in data.items() if val > min_ra]
            if target_node not in nonzero:
                nonzero += [target_node]

            subgraph = full_net.loc[nonzero,nonzero]
            friendly = friendlyNet(subgraph.values)
            friendly.NodeNames = nonzero

            if score_model == "LV":
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
            elif score_model == "AntLV":
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.antagonistic_lotka_volterra_score(target_node,numtrials = odeTrials)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
            elif score_model == "Replicator":
                if isinstance(odeTrials,int):
                    net_test_scores[indx] = friendly.replicator_score(target_node,numtrials = odeTrials)
                else:
                    net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
            elif score_model == "NodeBalance":
                net_test_scores[indx] = friendly.node_balanced_score(target_node)
            elif score_model == "Stochastic":
                net_test_scores[indx] = friendly.stochastic_score(target_node)
            elif score_model == "Composite":
                net_test_scores[indx] = friendly.score_node(target_node,odeTrials = odeTrials)["Composite"]
            indx += 1


        return sum(net_test_scores[net_scores == 1]) + sum(1-net_test_scores[net_scores == 0])

def score_single(sample,full_net,target_node, score_model, min_ra = 10**-6, odeTrials = None):

        net_scores = sample[0]
        data = sample[1]
        nonzero = [ky for ky,val in data.items() if val > min_ra]
        if target_node not in nonzero:
            nonzero += [target_node]

        wh = [np.where(np.array(fn_index) == nd)[0][0] for nd in nonzero]
        subgraph = full_net.loc[nonzero,nonzero]
        friendly = friendlyNet(subgraph.values)
        friendly.NodeNames = nonzero


        if score_model == "LV":
            if isinstance(odeTrials,int):
                net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = odeTrials)
            else:
                net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
        elif score_model == "AntLV":
            if isinstance(odeTrials,int):
                net_test_scores = friendly.antagonistic_lotka_volterra_score(target_node,numtrials = odeTrials)
            else:
                net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
        elif score_model == "Replicator":
            if isinstance(odeTrials,int):
                net_test_scores[indx] = friendly.replicator_score(target_node,numtrials = odeTrials)
            else:
                net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
        elif score_model == "NodeBalance":
            net_test_scores = friendly.node_balanced_score(target_node)
        elif score_model == "Stochastic":
            net_test_scores = friendly.stochastic_score(target_node)
        elif score_model == "Composite":
            net_test_scores = friendly.score_node(target_node,odeTrials = odeTrials)["Composite"]


        return net_scores,net_test_scores
