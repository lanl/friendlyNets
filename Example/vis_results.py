import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_binary(scores,aurocs,saveloc = "",experimentName = None, networkName=None, show = True):

    if len(saveloc):
        if not os.path.isdir(saveloc):
            os.makedirs(saveloc)

    font = {'weight' : 'bold',
        'size'   : 18}

    plt.rc('font', **font)

    melted_scores = scores.melt(id_vars = ["Score"], value_vars = ["LV","AntLV","Replicator","NodeBalance","Stochastic","Composite"])
    melted_scores["Engrafter"] = melted_scores["Score"].astype(bool)
    fig1,ax = plt.subplots(figsize = (15,10))

    sns.boxplot(x="variable", y="value",
                hue="Engrafter",
                data=melted_scores,ax = ax)
    if experimentName != None and networkName != None:
        plt.title("Test Score Distribution for {0}, {1} network".format(experimentName,networkName))
    else:
        plt.title("Test Score Distribution")
    plt.savefig(os.path.join(saveloc,"ts_distribution.png"))
    if show:
        plt.show()
    plt.close()

    fig,ax = plt.subplots(figsize = (15,10))
    sns.barplot(x = pd.Series(aurocs).index, y = pd.Series(aurocs),ax = ax)
    if experimentName != None and networkName != None:
        plt.title("AUROC Values for {0}, {1} network".format(experimentName,networkName))
    else:
        plt.title("AUROC Values")
    plt.savefig(os.path.join(saveloc,"aurocs.png"))
    if show:
        plt.show()
    plt.close()

def plot_continuous(scores,kendallval,spearmanval,saveloc = "",experimentName = None, networkName=None,show = True):

    if len(saveloc):
        if not os.path.isdir(saveloc):
            os.makedirs(saveloc)

    font = {'weight' : 'bold',
        'size'   : 18}

    plt.rc('font', **font)

    scores.rename({"Score":"Actual Engraftment"}, axis = 1, inplace = True)

    fig,ax = plt.subplots(2,3,figsize = (20,10),tight_layout = True)
    sns.scatterplot(x = "Actual Engraftment", y = "LV", data = scores,ax = ax[0,0])
    sns.scatterplot(x = "Actual Engraftment", y = "AntLV", data = scores,ax = ax[0,1])
    sns.scatterplot(x = "Actual Engraftment", y = "Replicator", data = scores,ax = ax[0,2])
    sns.scatterplot(x = "Actual Engraftment", y = "NodeBalance", data = scores,ax = ax[1,0])
    sns.scatterplot(x = "Actual Engraftment", y = "Stochastic", data = scores,ax = ax[1,1])
    sns.scatterplot(x = "Actual Engraftment", y = "Composite", data = scores,ax = ax[1,2])
    if experimentName != None and networkName != None:
        plt.suptitle("Actual Engraftment vs. Test Score for {0}, {1} network".format(experimentName,networkName))
    else:
        plt.suptitle("Actual Engraftment vs. Test Score")
    plt.savefig(os.path.join(saveloc,"sample_scatter.png"))
    if show:
        plt.show()
    plt.close()

    fig,ax = plt.subplots(figsize = (15,10))
    sns.barplot(x = pd.Series(kendallval).index, y = pd.Series(kendallval),ax = ax)
    if experimentName != None and networkName != None:
        plt.suptitle("Kendall Rank Correlation for {0}, {1} network".format(experimentName,networkName))
    else:
        plt.suptitle("Kendall Rank Correlation")
    plt.savefig(os.path.join(saveloc,"kendall_vals.png"))
    if show:
        plt.show()
    plt.close()

    fig,ax = plt.subplots(figsize = (15,10))
    sns.barplot(x = pd.Series(spearmanval).index, y = pd.Series(spearmanval),ax = ax)
    if experimentName != None and networkName != None:
        plt.suptitle("Spearman Rank Correlation for {0}, {1} network".format(experimentName,networkName))
    else:
        plt.suptitle("Spearman Rank Correlation")
    plt.savefig(os.path.join(saveloc,"pearson_vals.png"))
    if show:
        plt.show()
    plt.close()
