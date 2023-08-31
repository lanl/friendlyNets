import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
import os
import cobra as cb

sys.path.append(os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets"))
from make_gem_network import *


if __name__=="__main__":

    datapth = "b_longum_data"

    model_list = os.path.join(datapth,"model_list.csv")

    media_fl = "EU_average_AGORA.tsv"
    media_pth = os.path.join(os.path.expanduser("~"),"Documents","probiotic_design","friendlyNets","translate_agora_media",media_fl)


    rac_val = 35

    default_inflow = 10

    mu = 0.04

    phi = 0.1

    chunk_size = 100

    pair_growth,fba_growth,metadata = get_pairwise_growth(model_list,media_pth,rac=rac_val,mu=mu,phi=phi,default_inflow=default_inflow,chunk_size = chunk_size,silence_load_error=True)
 

    fld = os.path.join(datapth,"PairGrowth","{}_rac_{}".format(media_fl.split(".")[0],rac_val))
    Path(fld).mkdir(parents=True, exist_ok=True)
    pair_growth.to_csv(os.path.join(fld,"PairwiseGrowth.csv"))
    fba_growth.to_csv(os.path.join(fld,"FBAGrowth.csv"))
    with open(os.path.join(fld,"options.json"),'w') as fl:
        json.dump(metadata,fl)


