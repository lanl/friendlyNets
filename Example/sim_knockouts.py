import json
import pickle as pk
from turtle import numinput
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0,os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from friendlyNet import *
from score_net import *
import seaborn as sb
import matplotlib.pyplot as plt
from vis_results import *


if __name__ == "__main__":

    network_file = "pw_chemo_logRatio_thresh_top0.25.csv"
    level = "species" 


    full_net = pd.read_csv(network_file,index_col = 0)

    with open("target_nodes.json", 'r') as fl:
        targets = json.load(fl)


    with open("friendlySamples/upto_"+level+"/bifido_"+level+"_dict_binary.pk","rb") as fl:
        samples = pk.load(fl)
    
    target = "'Bifidobacterium_longum_infantis_157F_NC.mat'"

    ###Get a list of all the microbes present in all the samples
    all_nonzero = np.unique([list(smp[1].keys()) for smp in samples.values()])
    all_nonzero = all_nonzero[all_nonzero != target]
    
    knockout_results = pd.DataFrame(index = all_nonzero,columns = ["Present in All","Present in High","Present in Low","KO AUCROC","Original AUCROC", "AUCROC Difference", "Avg Score Difference","Avg Score Difference (High)","Average Score Difference (Low)"])
    all_sample_scores = pd.DataFrame(index = samples.keys(),columns = ["No KO"] + list(all_nonzero))

    

    aucroc_wi,sample_order_wi,test_scores_wi = score_light(samples,full_net,target,'b',"AntLV",keepscores = True)

    all_sample_scores.loc[sample_order_wi,"No KO"] = test_scores_wi

    numsamps = len(samples)

    high_samples = [ky for ky in samples.keys() if samples[ky][0] == 1]
    low_samples = [ky for ky in samples.keys() if samples[ky][0] == 0]

    nu = 0

    for ko in all_nonzero:

        try:
        
            #how many is it non-zero in, how many high, how many low
            numin = sum([smp[1][ko]>10**-6 for smp in samples.values()])/numsamps
            highin = sum([samples[ky][1][ko]>10**-6 for ky in high_samples])/len(high_samples)
            lowin = sum([samples[ky][1][ko]>10**-6 for ky in low_samples])/len(low_samples)
            
            aucroc_wo,sample_order_wo,test_scores_wo = score_light(samples,full_net,target,'b',"AntLV",KO = ko,keepscores = True)
            all_sample_scores.loc[sample_order_wo,ko] = test_scores_wo
            
            avg_diff = np.mean(all_sample_scores[ko] - all_sample_scores["No KO"])
            avg_high_diff = np.mean(all_sample_scores.loc[high_samples,ko] - all_sample_scores.loc[high_samples,"No KO"])
            avg_low_diff = np.mean(all_sample_scores.loc[low_samples,ko] - all_sample_scores.loc[low_samples,"No KO"])

            knockout_results.loc[ko] = [numin,highin,lowin,aucroc_wo,aucroc_wi,aucroc_wo-aucroc_wi,avg_diff,avg_high_diff,avg_low_diff]
            nu+=1

        except Exception as err:
            print("Exception on knockout {}:{}".format(nu,ko))
            print(type(err))    # the exception instance
            print(err.args)     # arguments stored in .args  
            nu+=1         


    all_sample_scores.to_csv(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results","koexperiments","sample_scores.csv"))
    knockout_results.to_csv(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results","koexperiments","summary.csv"))

