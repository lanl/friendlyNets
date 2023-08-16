import pandas as pd
import os
import json
import numpy as np

from make_gem_network import cocultures

def format_data(otu_table,**kwargs):

    """
    Formats a dataset (starting with OTU table) into the form ``dict{samplename:(score,dict{nodename:relative abundance})}``. Will also filter out OTUs by ID or name, so that OTUs left out of an interaction network can be removed (e.g. OTUs with no available GSM).

    :param otu_table: table of count or relative abundance data, index by OTU, columns corresponding to samples.
    :type otu_table: pandas.DataFrame

    :param sample_metadata: sample metadata use in determining the known score for a sample. If only predictions are desired, this can be omitted and all scores will be set to 1. Should be a table indexed by sample name (can also be a dict).
    :type sample_metadata: pandas.DataFrame

    :param score_col: column in sample_metadata indicating known score. Default "Score"
    :type score_col: str

    :param node_names: map from OTU table index to taxa names that match names in a known master network or match names of cobra models for network building.
    :type node_names: dict

    :param included_otus: list of otus to include in the data, all others will be excluded. Can be row names in otu table or corresponding value in name_names. Default include all otus
    :type included_otus: list

    :param scoretype: Either ``binary`` or ``continuous`` or ``infer`` for known scores for each sample. Default inferred from the sample metadata. **Will not** edit the scores to fit the given type.
    :type scoretype: str

    :return: Data formatted for easy use with friendlynet package (especially :py:func:`score_net <score_net.score_net>`)

    """

    sample_metadata = kwargs.get("sample_metadata",None)
    score_col = kwargs.get("score_col","Score")
    node_name = kwargs.get("node_names",None)
    included_otus = kwargs.get("included_otus",None)
    scoretype = kwargs.get("scoretype","Infer")

    if not isinstance(node_name,dict):
        node_name = dict([(i,i) for i in otu_table.index])

    if not hasattr(included_otus, "__len__"):#isinstance(included_otus,list):
        included_otus = otu_table.index


    experiment = {}
    coverage = {}

    otu_table = otu_table/otu_table.sum()

    samples = otu_table.columns

    for smp in samples:
        #Format the data vector as dictionary keyed by otu
        smpdata = {}
        missing = []
        for rw in otu_table.index:
            rwab = otu_table.loc[rw,smp]
            try:
                nmin = rwab.shape[0]
                print("{} appears {} times in table. Averaging.".format(rw,nmin))
            except:
                pass
            if (rw in included_otus) or (node_name[rw] in included_otus):
                if rw in node_name.keys():
                    smpdata[node_name[rw]] = rwab.mean()
                else:
                    smpdata[rw] = rwab.mean()
            else:
                if rwab.mean() > 0:
                    missing += [(rw,rwab.mean())]
        
        missing_arr = np.array([m[1] for m in missing])
        coverage[smp] = {"Coverage":1-sum(missing_arr), "NumberMissing":len(missing_arr),"MajorMissing":[m for m in missing if m[1]>0.8*max(missing_arr)],"AllMissing":missing}

        if isinstance(sample_metadata,pd.DataFrame):
            experiment[smp] = (sample_metadata.loc[smp,score_col],smpdata)
        elif isinstance(sample_metadata,dict):
            experiment[smp] = (sample_metadata[smp],smpdata)
        else:
            experiment[smp] = (1,smpdata)

    if scoretype.lower() == "infer":
        all_scores = np.array([s[0] for s in experiment.values()])
        if np.all(all_scores.round(9) == all_scores.astype(int)):
            return experiment,"binary",coverage
        else:
            return experiment,"continuous",coverage
    else:
        return experiment,scoretype,coverage
    



    