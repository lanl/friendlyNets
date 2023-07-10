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

def network_friendliness(experiment,full_net,target_node,models = None, min_ra = 10**-6, odeTrials = None):

    """

    This method computes the extent to which a set of sets of nodes are friendly to a particular node, using :py:obj:`friendlyNets <friendlyNets.friendlyNets>`.

    :param experiment: Dictionary of samples. Each should be a dictionary of abundances keyed by node names. (Also supports dict of tuples as in :py:func:`score_net <score_net.score_net>`
    :type experiment: dict[dict[float]] 
    :param full_net: Adjacency matrix of all interaction parameters between pairs of nodes in the entire experiment set.
    :type full_net: pandas dataframe
    :param target_node: name of node of interest. (must be in full_net, need not be in sample - will be added to induced subgraph)
    :type target_node: str
    :param models: list of models you wish to use for scoring. Leave as None for all 6. Options are "LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic".
    :type models: list
    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float
    :param odeTrials: number of ODE simulations in score estimation. If None, equal to number of non-zero taxa in a sample. Default None
    :type odeTrials: int

    :return: friendliness scores for each sample and each model in models.

    :rtype: pandas dataframe
    """

    if models == None:
        models = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic"]
        
    if "Composite" not in models:
        models = models + ["Composite"]
        
    net_scores = pd.DataFrame(index = experiment.keys(),columns = models)

    for ky,data in experiment.items():
        if not isinstance(data,dict): #in case it gets a tuple of (known score, data dict), we need to remove the known score part.
            data = data[1]

        nonzero = [kyy for kyy,val in data.items() if val > min_ra]
        if target_node not in nonzero:
            nonzero += [target_node]

        nonzero_rw = np.array(nonzero).astype(full_net.index.dtype)
        nonzero_cl = np.array(nonzero).astype(full_net.columns.dtype)

        subgraph = full_net.loc[nonzero_rw,nonzero_cl]
        friendly = friendlyNet(subgraph.values)
        friendly.NodeNames = nonzero

        fscores = friendly.score_node(target_node,odeTrials = odeTrials,scores=models)

        net_scores.loc[ky] = [fscores[col] for col in net_scores.columns[1:]]

    return net_scores


def score_net(experiment,full_net,target_node,scoretype,models = None, min_ra = 10**-6, odeTrials = None):

    """

    This method computes the extent to which a set of sets of nodes are friendly to a particular node, using :py:obj:`friendlyNets <friendlyNets.friendlyNets>`,
    and **compares** this with a known effect of the community, which can be binary (good/bad) or continuous (e.g. relative abundance at a later time point) to compute predictive performance

    :param experiment: Dictionary of samples, each a tuple with (known score,data). The data should be a dictionary of abundances keyed by node names.
    :type experiment: dict[tuple[float,dict]] 
    :param full_net: Adjacency matrix of all interaction parameters between pairs of nodes in the entire experiment set.
    :type full_net: pandas dataframe
    :param target_node: name of node of interest. (must be in full_net, need not be in sample - will be added to induced subgraph)
    :type target_node: str
    :param scoretype: Type of known score (`b` or `binary` for binary, `c` or `continuous` for continuous)
    :type scoretype: str
    :param models: list of models you wish to use for scoring. Leave as None for all 6. Options are "LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic".
    :type models: list
    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float
    :param odeTrials: number of ODE simulations in score estimation. If None, equal to number of non-zero taxa in a sample. Default None
    :type odeTrials: int

    :return: friendliness scores, predictive performance dictionary. If binary scoring, predictive performance is AUROC, ROC curves, and the mean AUROC.If continuous scoring, this is pearson correlation between friendliness score and known score, pearson p valule, kendall correlation, and kendall p value,spearman correlation, spearman p value. Correlation values are rescaled to [0,1] (from [-1,1]) to better match AUCROC scores.

    :rtype: tuple of pandas dataframe, {dict, dict, float} OR tuple of pandas dataframe, {dict,dict,dict,dict,dict,dict}

    """

    if models == None:
        models = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic"]
        
    if "Composite" not in models:
        models = models + ["Composite"]

    net_scores = pd.DataFrame(index = experiment.keys(),columns = ["KnownScore"] + models)

    for ky,sample in experiment.items():
        score = sample[0]
        data = sample[1]
        nonzero = [kyy for kyy,val in data.items() if val > min_ra]
        if target_node not in nonzero:
            nonzero += [target_node]

        nonzero_rw = np.array(nonzero).astype(full_net.index.dtype)
        nonzero_cl = np.array(nonzero).astype(full_net.columns.dtype)

        subgraph = full_net.loc[nonzero_rw,nonzero_cl]
        friendly = friendlyNet(subgraph.values)
        friendly.NodeNames = nonzero

        fscores = friendly.score_node(target_node,odeTrials = odeTrials,scores=models)

        net_scores.loc[ky] = [score] + [fscores[col] for col in net_scores.columns[1:]]

    if scoretype[0] == 'b':

        aurocs = {}
        roc_curves = {}
        for col in net_scores.columns[1:]:
            aurocs[col] = roc_auc_score(net_scores["KnownScore"].values.astype(float),net_scores[col].values.astype(float))
            roc_curves[col] = roc_curve(net_scores["KnownScore"].values.astype(float),net_scores[col].values.astype(float))

        return net_scores,{"AUCROC":aurocs,"ROC_Curves":roc_curves,"Mean_AUCROC":np.mean([val for val in aurocs.values()])}

    else:

        pearsonval_r = {}
        kendallval_r = {}
        spearmanval_r = {}
        pearsonp = {}
        kendallp = {}
        spearmanp = {}
        for col in net_scores.columns[1:]:
            pearsonval_r[col],pearsonp[col] = pearsonr(net_scores["KnownScore"].values.astype(float),net_scores[col].values.astype(float))
            kendallval_r[col],kendallp[col] = kendalltau(net_scores["KnownScore"].values.astype(float),net_scores[col].values.astype(float))
            spearmanval_r[col],spearmanp[col] = spearmanr(net_scores["KnownScore"].values.astype(float),net_scores[col].values.astype(float))

        pearsonval = dict([(ky,(val+1)/2) for ky,val in pearsonval_r.items()])
        kendallval = dict([(ky,(val+1)/2) for ky,val in kendallval_r.items()])
        spearmanval = dict([(ky,(val+1)/2) for ky,val in spearmanval_r.items()])


        return net_scores,{"Pearson":pearsonval,"PearsonP":pearsonp,"Kendall":kendallval,"KendallP":kendallp,"Spearman":spearmanval,"SpearmanP":spearmanp}

def score_light(experiment,full_net,target_node,scoretype, score_model,self_inhibit = 0, min_ra = 10**-6, odeTrials = None,lvshift = 0,cntbu = False,keepscores = False,KO = None):

    """

    This method computes the extent to which a set of sets of nodes are friendly to a particular node, using :py:obj:`friendlyNets <friendlyNets.friendlyNets>`,
    and compares this with a known effect of the community, which can be binary (good/bad) or continuous (e.g. relative abundance at a later time point). This version
    only computes a score for a single type of model, but offers more parameter flexibility, including Shift and Self Inhibition in the :py:func:`Lotka-Volterra system <friendlyNet.friendlyNet.lotka_volterra_system>`

    :param experiment: set of sets of nodes, each a tuple with (known score,data). The data should be a dictionary of abundances keyed by node names.
    :type experiment: dict[tuple[float,dict]] 
    :param full_net: Adjacency matrix of all interaction parameters between pairs of nodes in the entire experiment set.
    :type full_net: pandas dataframe
    :param target_node: name of node of interest. (must be in full_net, need not be in sample - will be added to induced subgraph)
    :type target_node: str
    :param scoretype: Type of known score (`b` for binary, `c` for continuous)
    :type scoretype: str
    :param score_model: Dynamical model to use for scoring. Choices LV, AntLV, InhibitLV, Replicator, NodeBalance, Stochastic, Composite, as detailed in :py:func:`score_node <friendlyNet.friendlyNet.score_node>`
    :type score_model: str
    :param self_inhibit: extent to which the model should include self inhibition - self interaction terms (:math:`A_{ii}`) will be set to negative ``self_inhibit``. Default 0
    :type self_inhibit: float
    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float
    :param odeTrials: number of ODE simulations in score estimation. If None, equal to number of non-zero taxa in a sample. Default None
    :type odeTrials: int
    :param lvshift: Uniform (subtracted) modifier to interactions. Lotka-Volterra parameters will be :py:obj:`Adjacency <friendlyNet.friendlyNet.Adjacency>` - ``shift``. Default 0
    :type lvshift: float
    :param cntbu: Whether or not to count the number of blow-ups in the simulations. Default False
    :type cntbu: bool
    :param keepscores: Wether or not to return the friendliness scores. If False, only returns the predictive performance.
    :type keepscores: bool
    :param KO: Knockout nodes to remove from data
    :type KO: str
    
    :return: evaluation of prediction, as AUCROC or (kendall, spearman), optionally count of ODE blowups, and optionally sample ordering, friendliness scores (in that order)
    :rtype: tuple float (or float,float), optional float, optional list[str], optional array[float]

    """



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

        nonzero_rw = np.array(nonzero).astype(full_net.index.dtype)
        nonzero_cl = np.array(nonzero).astype(full_net.columns.dtype)

        subgraph = full_net.loc[nonzero_rw,nonzero_cl]
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

# def score_binary(experiment,full_net,target_node, score_model, min_ra = 10**-6, odeTrials = None):

#         net_scores = np.empty(len(experiment),dtype = np.float64)
#         net_test_scores = np.empty(len(experiment),dtype = np.float64)
#         indx = 0

#         for ky,sample in experiment.items():
#             net_scores[indx] = sample[0]
#             data = sample[1]
#             nonzero = [ky for ky,val in data.items() if val > min_ra]
#             if target_node not in nonzero:
#                 nonzero += [target_node]

#             subgraph = full_net.loc[nonzero,nonzero]
#             friendly = friendlyNet(subgraph.values)
#             friendly.NodeNames = nonzero

#             if score_model == "LV":
#                 if isinstance(odeTrials,int):
#                     net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials)
#                 else:
#                     net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
#             elif score_model == "AntLV":
#                 if isinstance(odeTrials,int):
#                     net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 1)
#                 else:
#                     net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 1)
#             elif score_model == "Replicator":
#                 if isinstance(odeTrials,int):
#                     net_test_scores[indx] = friendly.replicator_score(target_node,numtrials = odeTrials)
#                 else:
#                     net_test_scores[indx] = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
#             elif score_model == "NodeBalance":
#                 net_test_scores[indx] = friendly.node_balanced_score(target_node)
#             elif score_model == "Stochastic":
#                 net_test_scores[indx] = friendly.stochastic_score(target_node)
#             elif score_model == "Composite":
#                 net_test_scores[indx] = friendly.score_node(target_node,odeTrials = odeTrials)["Composite"]
#             indx += 1


#         return sum(net_test_scores[net_scores == 1]) + sum(1-net_test_scores[net_scores == 0])

# def score_single(sample,full_net,target_node, score_model, min_ra = 10**-6, odeTrials = None):

#         net_scores = sample[0]
#         data = sample[1]
#         nonzero = [ky for ky,val in data.items() if val > min_ra]
#         if target_node not in nonzero:
#             nonzero += [target_node]

#         wh = [np.where(np.array(fn_index) == nd)[0][0] for nd in nonzero]
#         subgraph = full_net.loc[nonzero,nonzero]
#         friendly = friendlyNet(subgraph.values)
#         friendly.NodeNames = nonzero


#         if score_model == "LV":
#             if isinstance(odeTrials,int):
#                 net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = odeTrials)
#             else:
#                 net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
#         elif score_model == "AntLV":
#             if isinstance(odeTrials,int):
#                 net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = odeTrials,shift = 1)
#             else:
#                 net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0],shift = 1)
#         elif score_model == "Replicator":
#             if isinstance(odeTrials,int):
#                 net_test_scores[indx] = friendly.replicator_score(target_node,numtrials = odeTrials)
#             else:
#                 net_test_scores = friendly.lotka_volterra_score(target_node,numtrials = friendly.Adjacency.shape[0])
#         elif score_model == "NodeBalance":
#             net_test_scores = friendly.node_balanced_score(target_node)
#         elif score_model == "Stochastic":
#             net_test_scores = friendly.stochastic_score(target_node)
#         elif score_model == "Composite":
#             net_test_scores = friendly.score_node(target_node,odeTrials = odeTrials)["Composite"]


#         return net_scores,net_test_scores
