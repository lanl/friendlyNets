import json
import pickle as pk
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0,os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from friendlyNet import *
from sensitivity import *
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":
    network_file = "pw_chemo_logRatio_thresh_top0.25.csv"
    with open("target_nodes.json", 'r') as fl:
            targets = json.load(fl)
    full_net = pd.read_csv(network_file,index_col = 0)

    level = "species"
    with open("friendlySamples/upto_"+level+"/bifido_"+level+"_dict_binary.pk","rb") as fl:
                bifido_samples = pk.load(fl)
    now=datetime.now()

    try:
        saveloc = os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results",sys.argv[1])
    except:
        saveloc = os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results","bifido_sensitivity_shift1_"+now.strftime("%a%b%d_%H%M"))

    Ntri = 1#100

    for target_node in ["'Bifidobacterium_longum_infantis_157F_NC.mat'"]: # in targets["Bifido"][:1]:

        save_folder = os.path.join(saveloc,target_node.split(".")[0].split("_")[-1])
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        done_already = os.listdir(save_folder)

        mean_sense = pd.DataFrame(np.zeros_like(full_net.values),index = full_net.index, columns = full_net.columns)
        mean_sense_high = pd.DataFrame(np.zeros_like(full_net.values),index = full_net.index, columns = full_net.columns)
        mean_sense_low = pd.DataFrame(np.zeros_like(full_net.values),index = full_net.index, columns = full_net.columns)

        num_samp = len(bifido_samples)
        numH = len([smp for smp in bifido_samples.values() if smp[0] == 1])
        numL = len([smp for smp in bifido_samples.values() if smp[0] == 0])

        for ky,sample in bifido_samples.items():
            succ = True
            nonzero = [ky for ky,val in sample[1].items() if val>10**-6]
            if target_node not in nonzero:
                nonzero += [target_node]

            if "sensitivities_{}.csv".format(ky) in done_already:
                print("Loading {}".format(ky))
                sense = pd.read_csv(os.path.join(save_folder,"sensitivities_{}.csv".format(ky)),index_col = 0)

            else:         
                print("Solving {}".format(ky))
                subgraph = full_net.loc[nonzero,nonzero].values

                fnet = friendlyNet(subgraph)
                fnet.NodeNames = nonzero
                
                try:
                    weights = np.array([1.5**i for i in range(40)])
                    weights = weights/sum(weights)
                    sense = get_all_sensitivity_single_trajectory(target_node,fnet,mxTime = 1000,shift=1,weights = weights)#get_all_sensitivity(target_node,fnet,numtrials = Ntri,mxTime=1000,nj = 1,shift=1)
                    sense = pd.DataFrame(sense.T,index = fnet.NodeNames,columns = fnet.NodeNames)
                    sense.to_csv(os.path.join(save_folder,"sensitivities_{}.csv".format(ky)))
                except:
                    print("{} Failed".format(ky))
                    succ = False
          
            if succ:
                mean_sense.loc[sense.index,sense.columns] += sense.values/num_samp
                if sample[0] == 1:
                    mean_sense_high.loc[sense.index,sense.columns] += sense.values/numH
                else:
                    mean_sense_low.loc[sense.index,sense.columns] += sense.values/numL



        mean_sense.to_csv(os.path.join(save_folder,"mean_sensitivity.csv"))
        mean_sense_high.to_csv(os.path.join(save_folder,"mean_sensitivity_highSamples.csv"))
        mean_sense_low.to_csv(os.path.join(save_folder,"mean_sensitivity_lowSamples.csv"))

