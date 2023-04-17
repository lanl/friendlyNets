import numpy as np
import itertools as itert
import cobra as cb
import pandas as pd
import numbers

from steadycomX import format_models,set_media,steadyComXLite

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
    
    :param models: a dictionary of cobra models
    :type model: dict[GSM]
    
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

    :param keep_fluxes: If True, the new media will include the fluxes from the models previous media that do not appear in media. Default False.
    :type keep_fluxes: bool

    :param mu: specific community growth rate
    :type mu: float

    :param uptake: Upper bound of community uptake of metabolites 
    :type uptake: array

    :param phi: forced metabolite leak
    :type phi: float

    :param rac: Intracellular flux budget for RAC
    :type rac: float 

    :param experiment: set of sets of nodes, each a tuple with (known score,data). The data should be a dictionary of abundances keyed by names of models (same as dict keys of models). Used to determine co-occurrence
    :type experiment: dict[tuple[float,dict]] 

    :param min_ra: cutoff to use for presence/abscence of a node in a sample. Default 10**-6
    :type min_ra: float

    :return: table of simulated co-culture results. Row/Columns are index so Table.loc[row,col] = growth of row in coculture with col
    :rtype: pandas DF

    """
    media = kwargs.get("media",100)
    experiment = kwargs.get("experiment")
    min_ra = kwargs.get("min_ra")
    def_infl = kwargs.get("infl_default",10)
    IDtype = kwargs.get("IDtype","fullName")
    fluxcol = kwargs.get("fluxcol","fluxValue")
    compartmenttag = kwargs.get("compartmenttag","_e0")
    target_models = kwargs.get("target_models",[])

    if not hasattr(target_models, "__len__"):
        target_models = [target_models]

    if not isinstance(compartmenttag,dict):
        compartmenttag = dict([(mod,compartmenttag) for mod in cobra_models.keys()])

    model_parameters,metabolites = format_models(cobra_models,compartmenttag)

    metabolites_l = [m.lower() for m in metabolites]

    if isinstance(media,numbers.Number):
        U = media*np.ones(len(metabolites))
    elif isinstance(media,dict):
        U = def_infl*np.ones(len(metabolites))
        for ky,val in media.items():
            U[list(metabolites_l).index(ky.lower())] = val
    elif isinstance(media,pd.DataFrame):
        U = def_infl*np.ones(len(metabolites))
        for rw in media.index:
            if media.loc[rw,IDtype].lower() in metabolites_l:
                U[list(metabolites_l).index(media.loc[rw,IDtype].lower())] = media.loc[rw,fluxcol]

    kwargs["uptake"] = U

    if isinstance(experiment,pd.DataFrame):
        needmsk = check_co_occ(experiment,min_ra=min_ra)
    else:
        needmsk = pd.DataFrame(index = cobra_models.keys(),columns = cobra_models.keys()).fillna(True)

    for tg in target_models:
        needmsk.loc[tg] = True
        needmsk[tg] = True
    
    interaction_parameters = pd.DataFrame(index = cobra_models.keys(),columns = cobra_models.keys()).fillna(0.0)

    for col in interaction_parameters.columns:
        for row in interaction_parameters.index:
            if needmsk.loc[row,col]:
                biomasses = steadyComXLite({col:model_parameters[col],row:model_parameters[row]},**kwargs)
                interaction_parameters.loc[row,col] = biomasses[row]
                interaction_parameters.loc[col,row] = biomasses[col]
                needmsk.loc[col,row] = False

    return interaction_parameters