import pandas as pd
import os
import numpy as np
import sys
import json
import pickle as pk
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from score_net import score_net


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

    
###########

    experiment_dir = os.path.join("b_longum_data","Formatted","Baseline")

    rfld = os.path.join("results","Baseline")
    Path(rfld).mkdir(parents=True, exist_ok=True)


    testmodels = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic","Composite"]

    Path(os.path.join(rfld,"Predictions")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rfld,"ROC_Curves")).mkdir(parents=True, exist_ok=True)
    auc_roc_df = pd.DataFrame(columns=testmodels)
    with open(os.path.join(experiment_dir,"friendlySamples.json")) as fl:
        exp_di = json.load(fl)
    exp_data = exp_di["Data"]
    targ_node = exp_di["TargetNode"]
    scr_type = exp_di["ScoreType"]
    pred_df, scor = score_net(exp_data,full_network,targ_node,scr_type,odeTrials=ode_trials,models=testmodels)
    pred_df.to_csv(os.path.join(rfld,"Predictions","b_longum.csv"))
    auc_roc_df.loc["B.Longum"] = pd.Series(scor["AUCROC"])
    for md in testmodels:
        fpr,tpr,thr = scor["ROC_Curves"][md]
        fig,ax = plt.subplots(figsize = (5,5))
        ax.step(fpr,tpr)
        ax.plot([0,1],[0,1],':')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.savefig(os.path.join(rfld,"ROC_Curves","{}.png".format(md)))
        plt.close()
        with open(os.path.join(rfld,"ROC_Curves","{}.pk".format(md)),'wb') as pfl:
            pk.dump(scor["ROC_Curves"][md],pfl)
                    
    auc_roc_df.to_csv(os.path.join(rfld,"auc_roc.csv"))

###########

    experiment_dir = os.path.join("b_longum_data","Formatted","Treatment")

    rfld = os.path.join("results","Treatment")
    Path(rfld).mkdir(parents=True, exist_ok=True)


    testmodels = ["LV","InhibitLV","AntLV","Replicator","NodeBalance","Stochastic","Composite"]

    Path(os.path.join(rfld,"Predictions")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(rfld,"ROC_Curves")).mkdir(parents=True, exist_ok=True)
    auc_roc_df = pd.DataFrame(columns=testmodels)
    with open(os.path.join(experiment_dir,"friendlySamples.json")) as fl:
        exp_di = json.load(fl)
    exp_data = exp_di["Data"]
    targ_node = exp_di["TargetNode"]
    scr_type = exp_di["ScoreType"]
    pred_df, scor = score_net(exp_data,full_network,targ_node,scr_type,odeTrials=ode_trials,models=testmodels)
    pred_df.to_csv(os.path.join(rfld,"Predictions","b_longum.csv"))
    auc_roc_df.loc["B.Longum"] = pd.Series(scor["AUCROC"])
    for md in testmodels:
        fpr,tpr,thr = scor["ROC_Curves"][md]
        fig,ax = plt.subplots(figsize = (5,5))
        ax.step(fpr,tpr)
        ax.plot([0,1],[0,1],':')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        plt.savefig(os.path.join(rfld,"ROC_Curves","{}.png".format(md)))
        plt.close()
        with open(os.path.join(rfld,"ROC_Curves","{}.pk".format(md)),'wb') as pfl:
            pk.dump(scor["ROC_Curves"][md],pfl)
                    
    auc_roc_df.to_csv(os.path.join(rfld,"auc_roc.csv"))