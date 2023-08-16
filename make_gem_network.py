import numpy as np
import itertools as itert
import cobra as cb
import pandas as pd
import numbers
import json
import contextlib
import sys

from steadycomX import format_models,set_media,steadyComXLite


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostderr():
    save_stdout = sys.stderr
    sys.stderr = DummyFile()
    yield
    sys.stderr = save_stdout


    # min_ra = kwargs.get("min_ra")
    # def_infl = kwargs.get("default_inflow",10)
    # IDtype = kwargs.get("IDtype","fullName")
    # fluxcol = kwargs.get("fluxcol","fluxValue")
    # compartmenttag = kwargs.get("compartmenttag","_e0")
    # target_models = kwargs.get("target_models",[])
    # rows = kwargs.get("rows",[])
    # cols = kwargs.get("cols",[])

def get_pairwise_growth(cobra_model_list,media_fl,**kwargs):

    """

    Creates interaction network for list of cobra models 
    
    :param cobra_model_list: path to .csv file with info on GEM location. 
    :type cobra_model: str

    :param media_fl: path to media .csv
    :type media_fl: str

    :param experiment: path .json file containing set of sets of nodes, each a tuple with (known score,data). The data should be a dictionary of abundances keyed by names of models (same as dict keys of models). Used to determine co-occurrence. All pairs computed if not given.
    :type experiment: str
    
    :param silence_load_error: If true, suppresses error output in loading models. Default True.
    :type silence_load_error: bool

    :param IDtype: How the MODEL ids the metabolites. Should be a column of the media table. Default "fullName"
    :type IDtype: str

    :param compartmenttag: how the MODEl tags exchanged metabolites (e.g. _e0 for a modelSEED model). Default "_e0"
    :type compartmenttag: str

    :param fluxcol: column in media table with flux bound. Default "fluxValue"
    :type fluxcol: str

    :param keep_fluxes: If True, the new media will include the fluxes from the models previous media that do not appear in media. Default True.
    :type keep_fluxes: bool

    :param mu: specific community growth rate. Default 0.4
    :type mu: float

    :param phi: forced metabolite leak. Default 0.1
    :type phi: float

    :param rac: Intracellular flux budget for RAC. Default 100.
    :type rac: float 

    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float

    :return: table of simulated co-culture results. Row/Columns are index so Table.loc[row,col] = growth of row in coculture with col as well as FBA solution for each model. Growth of each model with standard FBA. Parameters used.
    :rtype: pandas.DataFrame, pandas.Series, dict

    """

    suppress_load_error = kwargs.get("silence_load_error",True)

    model_list = pd.read_csv(cobra_model_list,index_col=0)
    models_available = model_list[~model_list["ModelPath"].isna()]

    models_available.index = models_available.index.astype(int)

    experiment_fls = kwargs.get("experiment",[])
    if type(experiment_fls) is str:
        experiment_fls = [experiment_fls]


    experiments = []
    targets = []
    # target_mods = {}
    if len(experiment_fls):
        for exfl in experiment_fls:
            with open(exfl) as fl:
                experiment = json.load(fl)
            target = int(experiment["TargetNode"])
            if target not in targets:
                targets += [target]
                experiments += [experiment["Data"]]




    chsize = kwargs.get("chunk_size",100)
    chunks = [models_available.index[i*chsize:i*chsize+chsize] for i in range(int(len(models_available)/chsize)+1)]


    media = pd.read_csv(media_fl,sep = '\t',index_col= 0)

    pairwise_growth = pd.DataFrame(index = models_available.index,columns = models_available.index)

    fba_growth = pd.Series(index = models_available.index)


    diag_kws = kwargs.copy()
    diag_kws["experiments"] = experiments
    diag_kws["media"] = media
    diag_kws["target_models"] = targets

    off_diag_kws = kwargs.copy()
    off_diag_kws["experiments"] = experiments
    off_diag_kws["media"] = media
    off_diag_kws["target_models"] = targets

    for ci,chnk in enumerate(chunks):
        chnk_mods = {}
        for tid in chnk:
            if suppress_load_error:
                with nostderr():
                    mod =  cb.io.read_sbml_model(models_available.loc[tid,"ModelPath"])
            else:
                mod =  cb.io.read_sbml_model(models_available.loc[tid,"ModelPath"])
            chnk_mods[tid] = mod
            set_media(mod,media=media,keep_fluxes = True) 
            fba_growth.loc[tid] = mod.slim_optimize()
        # chnk_mods.update(target_mods)
        mes = "Diagonal chunk {}/{}".format(ci+1,len(chunks))
        diag_blk,_ = cocultures(chnk_mods,**diag_kws,message = mes)
        pairwise_growth.loc[chnk,chnk] = diag_blk
        for ci2 in range(ci+1,len(chunks)):
            chnk2 = chunks[ci2]
            chnk2_mods = {}
            for tid in chnk2:
                if suppress_load_error:
                    with nostderr():
                        chnk2_mods[tid] = cb.io.read_sbml_model(models_available.loc[tid,"ModelPath"])
                else:
                    chnk2_mods[tid] = cb.io.read_sbml_model(models_available.loc[tid,"ModelPath"])
            off_diag_kws["rows"] = chnk
            off_diag_kws["cols"] = chnk2
            mes = mes + "(Off-diagonal chunk {}/{})".format(ci2+1,len(chunks))
            offblk,offblk_T = cocultures({**chnk_mods,**chnk2_mods},**off_diag_kws,message = mes)
            pairwise_growth.loc[chnk,chnk2] = offblk
            pairwise_growth.loc[chnk2,chnk] = offblk_T

    metadata = {"Media File":media_fl,"TargetModel":"{}".format(targets)}
    metadata = {**metadata,**kwargs}

    return pairwise_growth,fba_growth,metadata


def check_co_occ(experiment,min_ra=10**-6):
    """

    Check which nodes co-occurr in samples, so that we don't need to compute an interaction between those that don't
    
    :param experiment: set of sets of nodes, as a dictionary of tuples with (known score,data) keyed by sample identifier. The data should be a dictionary of abundances keyed by node names.
    :type experiment: dict[tuple[float,dict]] 
    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float

    :return: NxN array with bool indicating if the pair co-occures, and list of nodes giving ordering of this array.
    :rtype: array[bool],list[str]
    
    """

    nonzero = {}
    for ky,sample in experiment.items():
        data = sample[1]
        nonzero[ky] = [kyy for kyy,val in data.items() if val > min_ra]
    
    kyorder = list(nonzero.keys())

    all_nodes = np.unique([nd for ky in kyorder for nd in nonzero[ky]])

    #save in membership for each node as a binary string. Then we can do bitwise & and skip any 0s.

    all_memberships = ["".join([str(int(nd in nonzero[k])) for k in kyorder]) for nd in all_nodes]

    needmsk = np.array([int(pr[0],2) & int(pr[1],2) for pr in itert.product(all_memberships, repeat=2)]).reshape((len(all_nodes),len(all_nodes))).astype(bool)

    return pd.DataFrame(needmsk,index = all_nodes,columns = all_nodes)


def cocultures(cobra_models,**kwargs):

    """

    Computes steadyComX simulations, including setting media for models, for all pairs of models (optionally only co-occurring models).
    
    :param cobra_models: a dictionary of cobra models
    :type cobra_model: dict[GSM]

    :param rows: list of keys to be used for 1 half of each pair
    :type rows: list[str]

    :param cols: list of keys to be used for other half of each pair. If either or both is not provided, does allxall
    :type cols: list[str]
    
    :param media: table including flux bound and modelSEED metabolite ID
    :type media:

    :param target_models: model/models of interest. Computes parameters for all interactions with these models. Must be in cobra_models. Default empty list
    :type target_models: str or list

    :param IDtype: How the MODEL ids the metabolites. Should be a column of the media table.
    :type IDtype: str

    :param compartmenttag: how the MODEl tags exchanged metabolites (e.g. _e0 for a modelSEED model)
    :type compartmenttag: str

    :param fluxcol: column in media table with flux bound
    :type fluxcol: str

    :param keep_fluxes: If True, the new media will include the fluxes from the models previous media that do not appear in media. Default True.
    :type keep_fluxes: bool

    :param mu: specific community growth rate
    :type mu: float

    :param default_inflow: Default inflow of metabolites not listed in media 
    :type default_inflow: float

    :param phi: forced metabolite leak
    :type phi: float

    :param rac: Intracellular flux budget for RAC
    :type rac: float 

    :param experiments: (possibly list of) set of sets of nodes, each a tuple with (known score,data). The data should be a dictionary of abundances keyed by names of models (same as dict keys of models). Used to determine co-occurrence
    :type experiments: dict[tuple[float,dict]] or list of same

    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float

    :return: table of simulated co-culture results. Row/Columns are index so Table.loc[row,col] = growth of row in coculture with col
    :rtype: pandas DF

    """
    media = kwargs.get("media",100)
    experiments = kwargs.get("experiments")

    if type(experiments) is not list:
        experiments = [experiments]
    if len(experiments) == 0:
        experiments = [None]

    min_ra = kwargs.get("min_ra",10**-6)
    def_infl = kwargs.get("default_inflow",10)
    IDtype = kwargs.get("IDtype","fullName")
    fluxcol = kwargs.get("fluxcol","fluxValue")
    compartmenttag = kwargs.get("compartmenttag","_e0")
    target_models = kwargs.get("target_models",[])
    rows = kwargs.get("rows",[])
    cols = kwargs.get("cols",[])

    logging = kwargs.get("message","")

    if not len(rows)*len(cols):
        rows = cobra_models.keys()
        cols = cobra_models.keys()

    if not hasattr(target_models, "__len__"):
        target_models = [target_models]

    if not isinstance(compartmenttag,dict):
        compartmenttag = dict([(mod,compartmenttag) for mod in cobra_models.keys()])

    model_parameters,metabolites = format_models(cobra_models,compartmenttag)

    metabolites_l = [m.lower() for m in metabolites]


    keep_media = kwargs.get("keep_fluxes",True)

    if isinstance(media,numbers.Number):
        U = media*np.ones(len(metabolites))
    elif isinstance(media,dict):
        if keep_media:
            U = def_infl*np.ones(len(metabolites))
        else:
            U = def_infl*np.zeros(len(metabolites))
        for ky,val in media.items():
            U[list(metabolites_l).index(ky.lower())] = val
    elif isinstance(media,pd.DataFrame):
        if keep_media:
            U = def_infl*np.ones(len(metabolites))
        else:
            U = def_infl*np.zeros(len(metabolites))        
        for rw in media.index:
            if media.loc[rw,IDtype].lower() in metabolites_l:
                # print(media.loc[rw,IDtype].lower(),media.loc[rw,fluxcol])
                U[list(metabolites_l).index(media.loc[rw,IDtype].lower())] = media.loc[rw,fluxcol]
            # else:
            #     print(media.loc[rw,IDtype].lower())

    kwargs["uptake"] = U

    needmsks = []

    for experiment in experiments:
        # print(experiment)
        if isinstance(experiment,pd.DataFrame):
            needmsks += [check_co_occ(experiment,min_ra=min_ra)]
        else:
            needmsks += [pd.DataFrame(index = rows,columns = cols).fillna(True)]
    
    all_needinds = np.unique([ni for df in needmsks for ni in df.index])
    all_need_cols = np.unique([ni for df in needmsks for ni in df.columns])
    pdded_needmsks = [pd.DataFrame(df,index = all_needinds,columns = all_need_cols).fillna(0) for df in needmsks]
    needmsk = sum(pdded_needmsks).astype(bool)

    # print(rows)
    # print(cols)
    # print(needmsk)

    for mod in rows:
        if mod not in needmsk.index:
            needmsk.loc[mod] = False
    for mod in cols:
        if mod not in needmsk.columns:
            needmsk[mod] = False


    for tg in target_models:
        if tg in needmsk.index:
            needmsk.loc[tg] = True
        if tg in needmsk.columns:
            needmsk[tg] = True
    
    interaction_parameters = pd.DataFrame(index = rows,columns = cols).fillna(0.0)
    interaction_parameters_T = pd.DataFrame(index = cols,columns = rows).fillna(0.0)

    for i,col in enumerate(cols):
        for j,row in enumerate(rows):
            if needmsk.loc[row,col]:
                print("{} --- col {}: {}/{} row {}: {}/{}".format(logging,col,i+1,len(cols),row,j+1,len(rows)))
                if row == col:
                    biomasses = steadyComXLite({col:model_parameters[col]},**kwargs)
                    biomasses[row] = biomasses[col]
                else:
                    biomasses = steadyComXLite({col:model_parameters[col],row:model_parameters[row]},**kwargs)
                interaction_parameters.loc[row,col] = biomasses[row]
                if list(rows) == list(cols):
                    interaction_parameters.loc[col,row] = biomasses[col]
                else:
                    interaction_parameters_T.loc[col,row] = biomasses[col]
                needmsk.loc[col,row] = False

    return interaction_parameters,interaction_parameters_T