import numpy as np
import cobra as cb
import pandas as pd
import os
import sys
import json
from pathlib import Path


sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))

# import make_gem_network as gmn
# import steadycomX as sx
from format_data import format_data

if __name__=="__main__":


    data_pth = "b_longum_data"

    #Load the OTU Table
    bif_otu = pd.read_csv(os.path.join(data_pth,"otu_table.csv"),index_col = 0)

    #The table indexed by NCBI Taxa ID, and has a column for species names. We are going to use NCBI Taxa ID for node names, but save a mapping to species names for convenience.
    bif_otu["Name"].to_csv(os.path.join(data_pth,"SpNames.csv"))

    #The metada for the samles includes names that indicate timepoint and subject. 
    bif_metadata = pd.read_csv(os.path.join(data_pth,"metagenome_info.csv"),index_col = 0)
    #And there is a seperate data table indicating cell counts of our target (b.longum) strain 
    bif_cellcounts = pd.read_excel(os.path.join(data_pth,"bifido_cell_counts.xlsx")).fillna(0)

    max_cell_counts = bif_cellcounts[bif_cellcounts["Days"] > 52].max()

    #We seperate time and subject name and only take samples from the baseline time-point (one for each subject

    bif_metadata["Time"] = bif_metadata["SampleName"].apply(lambda s: s.split("_")[1])
    bif_metadata["Subject"] = bif_metadata["SampleName"].apply(lambda s: s.split("_")[0])

###################
## Baseline timepoint

    baseline_metadata = bif_metadata[bif_metadata["Time"] == "baseline"][["Time","Subject"]]
    #We can then use the cell count data to determine engraftment.
    baseline_metadata["MaxCells"] = np.empty(len(baseline_metadata))
    baseline_metadata["Engrafter"] = np.empty(len(baseline_metadata),dtype=bool)
    for rw in baseline_metadata.index:
        sub = "Subject {}".format(baseline_metadata.loc[rw,"Subject"])
        baseline_metadata.loc[rw,"MaxCells"] = max_cell_counts.loc[sub]
        baseline_metadata.loc[rw,"Engrafter"] = int(max_cell_counts.loc[sub] > 4)

    #We need restrict the sample to the genomes we have available to make models of.

    attempted_models = pd.read_csv(os.path.join(data_pth,"model_list.csv"),index_col =0 )
    available_models = attempted_models[~attempted_models["ModelPath"].isna()]

    Path(os.path.join(data_pth,"Formatted","Baseline")).mkdir(parents=True, exist_ok=True)

    formatted = format_data(bif_otu[baseline_metadata.index],sample_metadata = baseline_metadata,score_col = "Engrafter",included_otus = available_models.index)

    all_info = {"TargetNode":216816,"Data":formatted[0],"ScoreType":formatted[1],"SampleCoverage":formatted[2]}

    coverage_info = pd.DataFrame.from_dict(formatted[2]).T
    coverage_info["MaxMissing"] = coverage_info["MajorMissing"].apply(lambda li: max([ite[1] for ite in li]))

    coverage_info.to_csv(os.path.join(data_pth,"Formatted","Baseline","sampleCoverage.tsv"),sep = '\t')
    coverage_info[["Coverage","NumberMissing","MaxMissing"]].astype(float).describe().to_csv(os.path.join(data_pth,"Formatted","Baseline","sampleCoverageSummary.csv"))

    with open(os.path.join(data_pth,"Formatted","Baseline","friendlySamples.json"),'w') as fl: 
        json.dump(all_info,fl)

#####################
## Treatment timepoint

    treatment_metadata = bif_metadata[bif_metadata["Time"] == "treatment"][["Time","Subject"]]
    #We can then use the cell count data to determine engraftment.
    treatment_metadata["MaxCells"] = np.empty(len(treatment_metadata))
    treatment_metadata["Engrafter"] = np.empty(len(treatment_metadata),dtype=bool)
    for rw in treatment_metadata.index:
        sub = "Subject {}".format(treatment_metadata.loc[rw,"Subject"])
        treatment_metadata.loc[rw,"MaxCells"] = max_cell_counts.loc[sub]
        treatment_metadata.loc[rw,"Engrafter"] = int(max_cell_counts.loc[sub] > 4)

    #We need restrict the sample to the genomes we have available to make models of.

    attempted_models = pd.read_csv(os.path.join(data_pth,"model_list.csv"),index_col =0 )
    available_models = attempted_models[~attempted_models["ModelPath"].isna()]



    Path(os.path.join(data_pth,"Formatted","Treatment")).mkdir(parents=True, exist_ok=True)

    formatted = format_data(bif_otu[treatment_metadata.index],sample_metadata = treatment_metadata,score_col = "Engrafter",included_otus = available_models.index)

    all_info = {"TargetNode":216816,"Data":formatted[0],"ScoreType":formatted[1],"SampleCoverage":formatted[2]}

    coverage_info = pd.DataFrame.from_dict(formatted[2]).T
    coverage_info["MaxMissing"] = coverage_info["MajorMissing"].apply(lambda li: max([ite[1] for ite in li]))

    coverage_info.to_csv(os.path.join(data_pth,"Formatted","Treatment","sampleCoverage.tsv"),sep = '\t')
    coverage_info[["Coverage","NumberMissing","MaxMissing"]].astype(float).describe().to_csv(os.path.join(data_pth,"Formatted","Treatment","sampleCoverageSummary.csv"))

    with open(os.path.join(data_pth,"Formatted","Treatment","friendlySamples.json"),'w') as fl: 
        json.dump(all_info,fl)

