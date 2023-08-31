import pandas as pd
import os
import numpy as np
import sys
import json
import pickle as pk
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from score_net import score_light
import itertools as it


def safelog(x,sval = -10):
    if x>0:
        return np.log(x)
    else:
        return sval

def make_logratio_network(pairwise_growth,sval = -10):
    ##log(x/y) = log(x) - log(y), allowing us to be more careful with zeros - now log(0/0) will return 0. log(x/0) will be large positive and log(0/x) will be large negative.
    network = (pairwise_growth.applymap(lambda s: safelog(s,sval=sval)).T - np.diag(pairwise_growth.applymap(lambda s: safelog(s,sval=sval)))).T
    return network


if __name__ == "__main__":

    ode_trials = 10


    pairgrowthpth = os.path.join("b_longum_data","PairGrowth","EU_average_AGORA_rac_35","PairwiseGrowth.csv")
    pair_growth = pd.read_csv(pairgrowthpth,index_col = 0)
    pair_growth.columns = pair_growth.columns.astype(int)

    full_network = make_logratio_network(pair_growth)

    #####################

    experiment_dir = os.path.join("b_longum_data","Formatted","Baseline")

    rfld = os.path.join("resultsLV","Baseline")
    Path(rfld).mkdir(parents=True, exist_ok=True)

    selfinhibit = np.linspace(0,1,10)
    shift = np.linspace(-1,1,20)


    with open(os.path.join(experiment_dir,"friendlySamples.json")) as fl:
        exp_di = json.load(fl)
    exp_data = exp_di["Data"]
    targ_node = exp_di["TargetNode"]
    scr_type = exp_di["ScoreType"]

    auc_roc_df = pd.DataFrame(columns = shift,index=selfinhibit)

    samples = list(exp_data.keys())
    all_scores_df = pd.DataFrame(index = samples,columns = ["Shift: {}, Inhibit:{}".format(x[0],x[1]) for x in it.product(shift,selfinhibit)])


    for slf_inh in selfinhibit:
        for lvsh in shift:
            auroc,smpls,scrs = score_light(exp_data,full_network,targ_node, 'b',"LV",lvshift = lvsh,self_inhibit = slf_inh, min_ra = 10**-6, odeTrials = ode_trials,keepscores = True)
            auc_roc_df.loc[slf_inh,lvsh] = auroc
            all_scores_df.loc[smpls,"Shift: {}, Inhibit:{}".format(lvsh,slf_inh)] = scrs
    
    auc_roc_df.to_csv(os.path.join(rfld,"auc_roc_df.csv"))
    all_scores_df.to_csv(os.path.join(rfld,"predictions.csv"))
    
    ##########################

    experiment_dir = os.path.join("b_longum_data","Formatted","Treatment")

    rfld = os.path.join("resultsLV","Treatment")
    Path(rfld).mkdir(parents=True, exist_ok=True)

    selfinhibit = np.linspace(0,1,10)
    shift = np.linspace(-1,1,20)


    with open(os.path.join(experiment_dir,"friendlySamples.json")) as fl:
        exp_di = json.load(fl)
    exp_data = exp_di["Data"]
    targ_node = exp_di["TargetNode"]
    scr_type = exp_di["ScoreType"]

    auc_roc_df = pd.DataFrame(columns = shift,index=selfinhibit)

    samples = list(exp_data.keys())
    all_scores_df = pd.DataFrame(index = samples,columns = ["Shift: {}, Inhibit:{}".format(x[0],x[1]) for x in it.product(shift,selfinhibit)])


    for slf_inh in selfinhibit:
        for lvsh in shift:
            auroc,smpls,scrs = score_light(exp_data,full_network,targ_node, 'b',"LV",lvshift = lvsh,self_inhibit = slf_inh, min_ra = 10**-6, odeTrials = ode_trials,keepscores = True)
            auc_roc_df.loc[slf_inh,lvsh] = auroc
            all_scores_df.loc[smpls,"Shift: {}, Inhibit:{}".format(lvsh,slf_inh)] = scrs
    
    auc_roc_df.to_csv(os.path.join(rfld,"auc_roc_df.csv"))
    all_scores_df.to_csv(os.path.join(rfld,"predictions.csv"))
    
