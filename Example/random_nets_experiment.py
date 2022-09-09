import json
import pickle as pk
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path


sys.path.insert(0,os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from friendlyNet import *
from score_net import *
import seaborn as sb
import matplotlib.pyplot as plt
import time
import json5

# @article{maldonado2016stable,
#   title={Stable engraftment of Bifidobacterium longum AH1206 in the human gut depends on individualized features of the resident microbiome},
#   author={Maldonado-G{\'o}mez, Mar{\'\i}a X and Mart{\'\i}nez, In{\'e}s and Bottacini, Francesca and O’Callaghan, Amy and Ventura, Marco and van Sinderen, Douwe and Hillmann, Benjamin and Vangay, Pajau and Knights, Dan and Hutkins, Robert W and others},
#   journal={Cell host \& microbe},
#   volume={20},
#   number={4},
#   pages={515--526},
#   year={2016},
#   publisher={Elsevier}
# }

# Testing for Lactobacillus_plantarum
#     ## Data from
# @article{huang2021candidate,
#   title={Candidate probiotic Lactiplantibacillus plantarum HNU082 rapidly and convergently evolves within human, mice, and zebrafish gut but differentially influences the resident microbiome},
#   author={Huang, Shi and Jiang, Shuaiming and Huo, Dongxue and Allaband, Celeste and Estaki, Mehrbod and Cantu, Victor and Belda-Ferre, Pedro and V{\'a}zquez-Baeza, Yoshiki and Zhu, Qiyun and Ma, Chenchen and others},
#   journal={Microbiome},
#   volume={9},
#   number={1},
#   pages={1--17},
#   year={2021},
#   publisher={BioMed Central}
# }



if __name__ == "__main__":

    font = {'weight' : 'bold',
            'size'   : 18}

    plt.rc('font', **font)

    network_file = "pw_chemo_logRatio.csv"
    level = "species"
    top_percent = 0.25
    mins = float(sys.argv[1])

    net_folder = os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results","random_nets_{}_LR_RAC".format(top_percent))

    Path(net_folder).mkdir(parents=True,exist_ok=True)


    models = ["LV","AntLV","InhibitLV","Replicator","NodeBalance","Stochastic","Composite"]

    print("Saving to {}".format(net_folder))

    full_net_real = pd.read_csv(network_file,index_col = 0)
    fnarr = full_net_real.values
    netvals = fnarr.flatten()[fnarr.flatten() != 0]




    with open("target_nodes.json", 'r') as fl:
        targets = json.load(fl)

    

    with open("friendlySamples/upto_"+level+"/cdiff_"+ level + "_dict.pk","rb") as fl:
        cdiffsamples = pk.load(fl)
    with open("friendlySamples/upto_"+level+"/bifido_"+level+"_dict_binary.pk","rb") as fl:
        bifidosamples = pk.load(fl)
    with open("friendlySamples/upto_"+level+"/lacto_day0_human_"+level+".pk","rb") as fl:
        lactosamples = pk.load(fl)


    try:
        with open(os.path.join(net_folder,'cdiff_scores.json'),'r') as fl:
            cdiff_scores = json5.load(fl)
    except:
        cdiff_scores = dict([(mod,dict([(target,[]) for target in targets["CDiff"]])) for mod in models])#pd.DataFrame(columns =  models,index = targets["CDiff"])
    try:
        with open(os.path.join(net_folder,'bifido_scores.json'),'r') as fl:
            bifido_scores = json5.load(fl)
    except:
        bifido_scores = dict([(mod,dict([(target,[]) for target in targets["Bifido"]])) for mod in models])#pd.DataFrame(columns =  models,index = targets["Bifido"])   
    try:
        with open(os.path.join(net_folder,'lacto_H0_scores_s.json'),'r') as fl:
            lacto_H0_scores_s = json5.load(fl)
    except:
        lacto_H0_scores_s =  dict([(mod,dict([(target,[]) for target in targets["Lacto"]])) for mod in models])#pd.DataFrame(columns =  models,index = targets["Lacto"])
    try:
        with open(os.path.join(net_folder,'lacto_H0_scores_k.json'),'r') as fl:
            lacto_H0_scores_k = json5.load(fl)
    except:
        lacto_H0_scores_k = dict([(mod,dict([(target,[]) for target in targets["Lacto"]])) for mod in models]) #pd.DataFrame(columns =  models,index = targets["Lacto"])

    for mod in models:
        if mod not in cdiff_scores.keys():
            cdiff_scores[mod] = dict([(target,[]) for target in targets["CDiff"]])
        if mod not in bifido_scores.keys():
            bifido_scores[mod] = dict([(target,[]) for target in targets["Bifido"]])
        if mod not in lacto_H0_scores_s.keys():
            lacto_H0_scores_s[mod] = dict([(target,[]) for target in targets["Lacto"]])
        if mod not in lacto_H0_scores_k.keys():
            lacto_H0_scores_k[mod] = dict([(target,[]) for target in targets["Lacto"]])


    tm0 = time.time()

    tm = 0

    while tm < mins*60:

        full_arr = np.random.choice(netvals,fnarr.shape)
        edge_thresh = np.quantile(abs(full_arr),1-top_percent)
        vals = np.zeros_like(full_arr)
        vals[np.where(full_arr > edge_thresh)] = full_arr[np.where(full_arr > edge_thresh)]
        vals[np.where(full_arr < -edge_thresh)] = full_arr[np.where(full_arr < -edge_thresh)]
        full_net = pd.DataFrame(vals,columns = full_net_real.columns, index = full_net_real.index)

        
        for target in targets["CDiff"]:
            for mod in models:
                auroc = score_light(cdiffsamples,full_net,target,'b',mod)
                cdiff_scores[mod][target] += [float(auroc)]

        for target in targets["Bifido"]:
            for mod in models:
                auroc = score_light(bifidosamples,full_net,target,'b',mod)
                bifido_scores[mod][target] += [float(auroc)]

        for target in targets["Lacto"]:
            for mod in models:
                ken,spear = score_light(lactosamples,full_net,target,'c',mod)
                lacto_H0_scores_k[mod][target] += [float(ken)]
                lacto_H0_scores_s[mod][target] += [float(spear)]

        tm = time.time()-tm0

        print(tm/60)

    with open(os.path.join(net_folder,'cdiff_scores.json'),'w') as fl:
        json5.dump(cdiff_scores,fl)
    with open(os.path.join(net_folder,'bifido_scores.json'),'w') as fl:
        json5.dump(bifido_scores,fl)
    with open(os.path.join(net_folder,'lacto_H0_scores_s.json'),'w') as fl:
        json5.dump(lacto_H0_scores_s,fl)
    with open(os.path.join(net_folder,'lacto_H0_scores_k.json'),'w') as fl:
        json5.dump(lacto_H0_scores_k,fl)