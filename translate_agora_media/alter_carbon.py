import pandas as pd
import numpy as np
import os

if __name__=="__main__":
    

    starting_media_file = "EU_average_AGORA.tsv"
    starting_media = pd.read_csv(starting_media_file,index_col = 0,sep ='\t')


    carbon_sources = ['D-Maltose','Sucrose','D-Fructose','D-Glucose']

    doubled_df = starting_media.copy()
    halved_df = starting_media.copy()

    change_these = starting_media[[nm in carbon_sources for nm in starting_media["fullName"]]].index

    doubled_df.loc[change_these,["fluxValue","g_day"]] = 2*starting_media.loc[change_these,["fluxValue","g_day"]]
    halved_df.loc[change_these,["fluxValue","g_day"]] = 0.5*starting_media.loc[change_these,["fluxValue","g_day"]]

    doubled_df.to_csv("EU_average_AGORA_carbon_doubled.tsv",sep="\t")
    halved_df.to_csv("EU_average_AGORA_carbon_halved.tsv",sep="\t")