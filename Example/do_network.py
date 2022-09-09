import json
import pickle as pk
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

    network_file = sys.argv[1] #input("Network file: ")
    net_name =sys.argv[2]# input("Net Name: ")
    level =sys.argv[3]# input("Tax Level Matching: ")
    net_folder =sys.argv[4]# input("Save Folder Name: ")

    net_folder = os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","pb_design_results",net_folder)

    print("Saving to {}".format(net_folder))

    full_net = pd.read_csv(network_file,index_col = 0)

    with open("target_nodes.json", 'r') as fl:
        targets = json.load(fl)

    models = ["LV","AntLV","InhibitLV","Replicator","NodeBalance","Stochastic","Composite"]
    # all_scores = pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
    best_models = pd.DataFrame(index = ["C.difficile","B.longum (S)","B.longum (K)","L.plantarum Day 0 Human (S)","L.plantarum Day 0 Human (K)","L.plantarum Day 0 Animal (S)","L.plantarum Day 0 Animal (K)","L.plantarum Day 3 Human (S)","L.plantarum Day 3 Human (K)","L.plantarum Day 3 Animal (S)","L.plantarum Day 3 Animal (K)"],columns = models)


    with open("friendlySamples/upto_"+level+"/cdiff_"+ level + "_dict.pk","rb") as fl:
        samples = pk.load(fl)
    cdiff_scores = pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
    for target in targets["CDiff"]:
        id = target.split("_")[-1].split(".")[0]
        net_scores,aurocs,roc_curves,mean = score_net(samples,full_net,target,'b')
        print(id + " Scored")
        plot_binary(net_scores,aurocs,saveloc=os.path.join(net_folder,level,"cdiff",id),networkName=net_name,experimentName="C. difficile",show = False)
        # all_scores.loc["Cd"+id] = ["C. difficile",id,"AUROC"] + [aurocs[mod] for mod in models]
        cdiff_scores.loc["Cd_" + id + "_A"] = ["C. difficile",id,"AUROC"] + [aurocs[mod] for mod in models]
        net_scores.to_csv(os.path.join(net_folder,level,"cdiff",id,"sample_scores.csv"))
        for model in models:
            fig,ax = plt.subplots(figsize = (10,10))
            ax.plot(roc_curves[model][0],roc_curves[model][1],linewidth = 5)
            ax.plot(np.arange(0,2,0.1),np.arange(0,2,0.1),":",linewidth = 3)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            plt.savefig(os.path.join(net_folder,level,"cdiff",id,"{}_roc.png".format(model)))
            plt.close()

    cdiff_scores.to_csv(os.path.join(net_folder,level,"cdiff","cdiff_scores.csv"))
    cdiff_scores.to_latex(os.path.join(net_folder,level,"cdiff","cdiff_scores.tex"))
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(cdiff_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"cdiff","cdiff_scores_hm.png"))
    plt.close()
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(cdiff_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"cdiff","cdiff_scores.png"))
    plt.close()
    for mod in models:
        best_models.loc["C.difficile",mod] = "/".join(list(cdiff_scores[cdiff_scores[mod] == cdiff_scores.max()[mod]].index))

    bif_cont = False

    if bif_cont:
        with open("friendlySamples/upto_"+level+"/bifido_"+level+"_dict_cc.pk","rb") as fl:
            samples = pk.load(fl)
        bifido_scores = pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
        for target in targets["Bifido"]:
            id = target.split("_")[-1].split(".")[0]
            net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp = score_net(samples,full_net,target,'c')
            print(id + " Scored")
            plot_continuous(net_scores,kendallval,spearmanval,saveloc=os.path.join(net_folder,level,"bifido",id),experimentName = "B. longum", networkName=net_name,show = False)
            # all_scores.loc["BlS"+id] = ["B. longum",id,"Spearman"] + [spearmanval[mod] for mod in models]
            # all_scores.loc["BlK"+id] = ["B. longum",id,"Kendall"] + [kendallval[mod] for mod in models]
            bifido_scores.loc["Bl_" + id + "_S"] = ["B. longum",id,"Spearman"] + [spearmanval[mod] for mod in models]
            bifido_scores.loc["Bl_" + id + "_K"] = ["B. longum",id,"Kendall"] + [kendallval[mod] for mod in models]
            net_scores.to_csv(os.path.join(net_folder,level,"bifido",id,"sample_scores.csv"))
        bifido_scores.to_csv(os.path.join(net_folder,level,"bifido","bifido_scores.csv"))
        bifido_scores.to_latex(os.path.join(net_folder,level,"bifido","bifido_scores.tex"))
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(bifido_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"bifido","bfidio_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(bifido_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"bifido","bifido_scores.png"))
        plt.close()
        for mod in models:
            sdf = bifido_scores[bifido_scores["ScoreType"] == "Spearman"]
            kdf = bifido_scores[bifido_scores["ScoreType"] == "Kendall"]
            best_models.loc["B.longum (S)",mod] = "/".join(list(sdf[sdf[mod] == sdf.max()[mod]].index))
            best_models.loc["B.longum (K)",mod] = "/".join(list(kdf[kdf[mod] == kdf.max()[mod]].index))

    else:
        with open("friendlySamples/upto_"+level+"/bifido_"+level+"_dict_binary.pk","rb") as fl:
            samples = pk.load(fl)
        bifido_scores = pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
        for target in targets["Bifido"]:
            id = target.split("_")[-1].split(".")[0]
            net_scores,aurocs,roc_curves,mean = score_net(samples,full_net,target,'b')
            print(id + " Scored")
            plot_binary(net_scores,aurocs,saveloc=os.path.join(net_folder,level,"bifido",id),experimentName = "B. longum", networkName=net_name,show = False)
            bifido_scores.loc["Bl_" + id + "_A"] = ["B. longum",id,"AUROC"] + [aurocs[mod] for mod in models]
            net_scores.to_csv(os.path.join(net_folder,level,"bifido",id,"sample_scores.csv"))
            for model in models:
                fig,ax = plt.subplots(figsize = (10,10))
                ax.plot(roc_curves[model][0],roc_curves[model][1],linewidth = 5)
                ax.plot(np.arange(0,2,0.1),np.arange(0,2,0.1),":",linewidth = 3)
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                plt.savefig(os.path.join(net_folder,level,"bifido",id,"{}_roc.png".format(model)))
                plt.close()



        bifido_scores.to_csv(os.path.join(net_folder,level,"bifido","bifido_scores.csv"))
        bifido_scores.to_latex(os.path.join(net_folder,level,"bifido","bifido_scores.tex"))
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(bifido_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"bifido","bfidio_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(bifido_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"bifido","bifido_scores.png"))
        plt.close()
        for mod in models:
            best_models.loc["B.longum (S)",mod] = "/".join(list(bifido_scores[bifido_scores[mod] == bifido_scores.max()[mod]].index))

    with open("friendlySamples/upto_"+level+"/lacto_day0_human_"+level+".pk","rb") as fl:
        samples = pk.load(fl)
    lacto_H0_scores =  pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
    for target in targets["Lacto"]:
        id = target.split("_")[-1].split(".")[0]
        net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp = score_net(samples,full_net,target,'c')
        print(id + " Scored")
        plot_continuous(net_scores,kendallval,spearmanval,saveloc=os.path.join(net_folder,level,"lacto_H0",id),experimentName = "L. plantarum Human Day 0", networkName=net_name, show = False)
        # all_scores.loc["LpSH0"+id] = ["L. plantarum Human Day 0","Spearman"] + [spearmanval[mod] for mod in models]
        # all_scores.loc["LpKH0"+id] = ["L. plantarum Human Day 0","Kendall"] + [kendallval[mod] for mod in models]
        lacto_H0_scores.loc["Lp_"+ id + "_S_H0"] = ["L. plantarum Human Day 0",id,"Spearman"] + [spearmanval[mod] for mod in models]
        lacto_H0_scores.loc["Lp_"+ id + "_K_H0"] = ["L. plantarum Human Day 0",id,"Kendall"] + [kendallval[mod] for mod in models]
        net_scores.to_csv(os.path.join(net_folder,level,"lacto_H0",id,"sample_scores.csv"))
    lacto_H0_scores.to_csv(os.path.join(net_folder,level,"lacto_H0","lacto_H0_scores.csv"))
    lacto_H0_scores.to_latex(os.path.join(net_folder,level,"lacto_H0","lacto_H0_scores.tex"))
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(lacto_H0_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"lacto_H0","lacto_H0_scores_hm.png"))
    plt.close()
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(lacto_H0_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"lacto_H0","lacto_H0_scores.png"))
    plt.close()
    for mod in models:
        sdf = lacto_H0_scores[lacto_H0_scores["ScoreType"] == "Spearman"]
        kdf = lacto_H0_scores[lacto_H0_scores["ScoreType"] == "Kendall"]
        best_models.loc["L.plantarum Day 0 Human (S)",mod] = "/".join(list(sdf[sdf[mod] == sdf.max()[mod]].index))
        best_models.loc["L.plantarum Day 0 Human (K)",mod] = "/".join(list(kdf[kdf[mod] == kdf.max()[mod]].index))

    all_lacto = False
    if all_lacto:
        with open("friendlySamples/upto_"+level+"/lacto_day0_animal_"+level+".pk","rb") as fl:
            samples = pk.load(fl)
        lacto_A0_scores =  pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
        for target in targets["Lacto"]:
            id = target.split("_")[-1].split(".")[0]
            net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp = score_net(samples,full_net,target,'c')
            print(id + " Scored")
            plot_continuous(net_scores,kendallval,spearmanval,saveloc=os.path.join(net_folder,level,"lacto_A0",id),experimentName = "L. plantarum Animal Day 0", networkName=net_name, show = False)
            # all_scores.loc["LpSA0"+id] = ["L. plantarum Animal Day 0","Spearman"] + [spearmanval[mod] for mod in models]
            # all_scores.loc["LpKA0"+id] = ["L. plantarum Animal Day 0","Kendall"] + [kendallval[mod] for mod in models]
            lacto_A0_scores.loc["Lp_"+ id + "_S_A0"] = ["L. plantarum Animal Day 0",id,"Spearman"] + [spearmanval[mod] for mod in models]
            lacto_A0_scores.loc["Lp_"+ id + "_K_A0"] = ["L. plantarum Animal Day 0",id,"Kendall"] + [kendallval[mod] for mod in models]
            net_scores.to_csv(os.path.join(net_folder,level,"lacto_A0",id,"sample_scores.csv"))
        lacto_A0_scores.to_csv(os.path.join(net_folder,level,"lacto_A0","lacto_A0_scores.csv"))
        lacto_A0_scores.to_latex(os.path.join(net_folder,level,"lacto_A0","lacto_A0_scores.tex"))
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_A0_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_A0","lacto_A0_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_A0_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_A0","lacto_A0_scores.png"))
        plt.close()
        for mod in models:
            sdf = lacto_A0_scores[lacto_A0_scores["ScoreType"] == "Spearman"]
            kdf = lacto_A0_scores[lacto_A0_scores["ScoreType"] == "Kendall"]
            best_models.loc["L.plantarum Day 0 Animal (S)",mod] = "/".join(list(sdf[sdf[mod] == sdf.max()[mod]].index))
            best_models.loc["L.plantarum Day 0 Animal (K)",mod] = "/".join(list(kdf[kdf[mod] == kdf.max()[mod]].index))

        with open("friendlySamples/upto_"+level+"/lacto_day3_human_"+level+".pk","rb") as fl:
            samples = pk.load(fl)
        lacto_H3_scores =  pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
        for target in targets["Lacto"]:
            id = target.split("_")[-1].split(".")[0]
            net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp = score_net(samples,full_net,target,'c')
            print(id + " Scored")
            plot_continuous(net_scores,kendallval,spearmanval,saveloc=os.path.join(net_folder,level,"lacto_H3",id),experimentName = "L. plantarum Human Day 0", networkName=net_name, show = False)
            # all_scores.loc["LpSH3"+id] = ["L. plantarum Human Day 3","Spearman"] + [spearmanval[mod] for mod in models]
            # all_scores.loc["LpKH3"+id] = ["L. plantarum Human Day 3","Kendall"] + [kendallval[mod] for mod in models]
            lacto_H3_scores.loc["Lp_"+ id + "_S_H3"] = ["L. plantarum Human Day 3",id,"Spearman"] + [spearmanval[mod] for mod in models]
            lacto_H3_scores.loc["Lp_"+ id + "_K_H3"] = ["L. plantarum Human Day 3",id,"Kendall"] + [kendallval[mod] for mod in models]
            net_scores.to_csv(os.path.join(net_folder,level,"lacto_H3",id,"sample_scores.csv"))
        lacto_H3_scores.to_csv(os.path.join(net_folder,level,"lacto_H3","lacto_H3_scores.csv"))
        lacto_H3_scores.to_latex(os.path.join(net_folder,level,"lacto_H3","lacto_H3_scores.tex"))
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_H3_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_H3","lacto_H3_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_H3_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_H3","lacto_H3_scores.png"))
        plt.close()
        for mod in models:
            sdf = lacto_H3_scores[lacto_H3_scores["ScoreType"] == "Spearman"]
            kdf = lacto_H3_scores[lacto_H3_scores["ScoreType"] == "Kendall"]
            best_models.loc["L.plantarum Day 3 Human (S)",mod] = "/".join(list(sdf[sdf[mod] == sdf.max()[mod]].index))
            best_models.loc["L.plantarum Day 3 Human (K)",mod] = "/".join(list(kdf[kdf[mod] == kdf.max()[mod]].index))

        with open("friendlySamples/upto_"+level+"/lacto_day3_animal_"+level+".pk","rb") as fl:
            samples = pk.load(fl)
        lacto_A3_scores =  pd.DataFrame(columns = ["Experiment","GEM","ScoreType"] + models)
        for target in targets["Lacto"]:
            id = target.split("_")[-1].split(".")[0]
            net_scores,pearsonval,pearsonp,kendallval,kendallp,spearmanval,spearmanp = score_net(samples,full_net,target,'c')
            print(id + " Scored")
            plot_continuous(net_scores,kendallval,spearmanval,saveloc=os.path.join(net_folder,level,"lacto_A3",id),experimentName = "L. plantarum Animal Day 0", networkName=net_name, show = False)
            # all_scores.loc["LpSA3"+id] = ["L. plantarum Animal Day 3","Spearman"] + [spearmanval[mod] for mod in models]
            # all_scores.loc["LpKA3"+id] = ["L. plantarum Animal Day 3","Kendall"] + [kendallval[mod] for mod in models]
            lacto_A3_scores.loc["Lp_"+ id + "_S_A3"] = ["L. plantarum Animal Day 3",id,"Spearman"] + [spearmanval[mod] for mod in models]
            lacto_A3_scores.loc["Lp_"+ id + "_K_A3"] = ["L. plantarum Animal Day 3",id,"Kendall"] + [kendallval[mod] for mod in models]
            net_scores.to_csv(os.path.join(net_folder,level,"lacto_A3",id,"sample_scores.csv"))
        lacto_A3_scores.to_csv(os.path.join(net_folder,level,"lacto_A3","lacto_A3_scores.csv"))
        lacto_A3_scores.to_latex(os.path.join(net_folder,level,"lacto_A3","lacto_A3_scores.tex"))
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_A3_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_A3","lacto_A3_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (10,10))
        sns.heatmap(lacto_A3_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"lacto_A3","lacto_A3_scores.png"))
        plt.close()
        for mod in models:
            sdf = lacto_A3_scores[lacto_A3_scores["ScoreType"] == "Spearman"]
            kdf = lacto_A3_scores[lacto_A3_scores["ScoreType"] == "Kendall"]
            best_models.loc["L.plantarum Day 3 Animal (S)",mod] = "/".join(list(sdf[sdf[mod] == sdf.max()[mod]].index))
            best_models.loc["L.plantarum Day 3 Animal (K)",mod] = "/".join(list(kdf[kdf[mod] == kdf.max()[mod]].index))

    best_models.to_csv(os.path.join(net_folder,level,"best_models.csv"))
    best_models.to_latex(os.path.join(net_folder,level,"best_models.tex"))

    if all_lacto:
        all_scores = pd.concat([cdiff_scores,bifido_scores,lacto_H0_scores,lacto_A0_scores,lacto_H3_scores,lacto_A3_scores])
        all_scores.to_csv(os.path.join(net_folder,level,"all_scores.csv"))
        all_scores.to_latex(os.path.join(net_folder,level,"all_scores.tex"))
        fig,ax = plt.subplots(figsize = (20,20))
        sns.heatmap(all_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"all_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (20,20))
        sns.heatmap(all_scores[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"all_scores.png"))
        plt.close()

        #without lacto day 3
        woday3 = pd.concat([cdiff_scores,bifido_scores,lacto_H0_scores,lacto_A0_scores])
        woday3.to_csv(os.path.join(net_folder,level,"wo_day3.csv"))
        woday3.to_latex(os.path.join(net_folder,level,"wo_day3.tex"))
        fig,ax = plt.subplots(figsize = (20,20))
        sns.heatmap(woday3[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"wo_day3_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (20,20))
        sns.heatmap(woday3[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"wo_day3.png"))
        plt.close()

    #without lacto day 3/day 0 animal
    justH0 = pd.concat([cdiff_scores,bifido_scores,lacto_H0_scores])
    justH0.to_csv(os.path.join(net_folder,level,"human_scores.csv"))
    justH0.to_latex(os.path.join(net_folder,level,"human_scores.tex"))
    fig,ax = plt.subplots(figsize = (15,15))
    sns.heatmap(justH0[models], ax = ax, center = 0.5, cmap = 'coolwarm')
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"human_scores_hm.png"))
    plt.close()
    fig,ax = plt.subplots(figsize = (15,15))
    sns.heatmap(justH0[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"human_scores.png"))
    plt.close()

    #Best on average
    if all_lacto:
        max_mean_df = pd.DataFrame(columns = all_scores.columns)
        for df in [cdiff_scores,bifido_scores,lacto_H0_scores,lacto_A0_scores,lacto_H3_scores,lacto_A3_scores]:
            mxind = df.index[df.mean(axis = 1).argmax()]
            max_mean_df.loc[mxind] = df.loc[mxind]
        max_mean_df.to_csv(os.path.join(net_folder,level,"best_mean_scores.csv"))
        max_mean_df.to_latex(os.path.join(net_folder,level,"best_mean_scores.tex"))
        fig,ax = plt.subplots(figsize = (15,15))
        sns.heatmap(max_mean_df[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"best_mean_scores_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (15,15))
        sns.heatmap(max_mean_df[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"best_mean_scores.png"))
        plt.close()

        #Best on average without day 3
        max_mean_df_wo3 = pd.DataFrame(columns = all_scores.columns)
        for df in [cdiff_scores,bifido_scores,lacto_H0_scores,lacto_A0_scores]:
            mxind = df.index[df.mean(axis = 1).argmax()]
            max_mean_df_wo3.loc[mxind] = df.loc[mxind]
        max_mean_df_wo3.to_csv(os.path.join(net_folder,level,"best_mean_scores_wod3.csv"))
        max_mean_df_wo3.to_latex(os.path.join(net_folder,level,"best_mean_scores_wod3.tex"))
        fig,ax = plt.subplots(figsize = (15,15))
        sns.heatmap(max_mean_df_wo3[models], ax = ax, center = 0.5, cmap = 'coolwarm')
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"best_mean_scores_wod3_hm.png"))
        plt.close()
        fig,ax = plt.subplots(figsize = (15,15))
        sns.heatmap(max_mean_df_wo3[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
        plt.title("Score Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(net_folder,level,"best_mean_scores_wod3.png"))
        plt.close()

    #Best on average just human
    max_mean_df_jh = pd.DataFrame(columns = justH0.columns)
    for df in [cdiff_scores,bifido_scores,lacto_H0_scores]:
        mxind = df.index[df.mean(axis = 1).argmax()]
        max_mean_df_jh.loc[mxind] = df.loc[mxind]
    max_mean_df_jh.to_csv(os.path.join(net_folder,level,"best_mean_scores_jh.csv"))
    max_mean_df_jh.to_latex(os.path.join(net_folder,level,"best_mean_scores_jh.tex"))
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(max_mean_df_jh[models], ax = ax, center = 0.5, cmap = 'coolwarm')
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"best_mean_scores_jh_hm.png"))
    plt.close()
    fig,ax = plt.subplots(figsize = (10,10))
    sns.heatmap(max_mean_df_jh[models], ax = ax, center = 0.5, cmap = 'coolwarm', annot = True)
    plt.title("Score Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(net_folder,level,"best_mean_scores_jh.png"))
    plt.close()
