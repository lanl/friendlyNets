import numpy as np
import pandas as pd
import cobra as cb
import gurobipy as gb
import sys
import numbers

def steadyComX(models,**kwargs):

    """

    Computes steadyComX simulation, including setting media for models, and fancifies the output
    
    :param models: a dictionary of cobra models
    :type model: dict[GSM]
    
    :param media: number, dict, or table indicating community uptake flux bounds
    :type media: number/dict/table

    :param infl_default: default upper bound for metabolite inflow (e.g. for metabolites not found in given media). Default 0
    :type phi: float

    :param IDtype: How the MODELS id the metabolites. Should be a column of the media table.
    :type IDtype: str

    :param compartmenttag: how the MODElS tags exchanged metabolites (e.g. _e0 for a modelSEED model)
    :type compartmenttag: str

    :param fluxcol: column in media table with flux bound
    :type fluxcol: str

    :param keep_fluxes: If True, the new media will include the fluxes from the models previous media that do not appear in media. Default False.
    :type keep_fluxes: bool

    :param mu: specific community growth rate
    :type mu: float

    :param phi: forced metabolite leak
    :type phi: float

    :param rac: Intracellular flux budget for RAC
    :type rac: float 

    :param print_LP: Option to print the constraints of the linear program. default False
    :type print_LP: bool

    :return: Community FBA simulation result from steadycomX. Returns dictionary of biomasses, reaction fluxes, and environmental metabolite community outflow. Exchange reaction fluxes are labeled by metabolite and correspond to net flux out of the microbe. Internal fluxes are a list ordered according to S
    :rtype: dict

    """
    media = kwargs.get("media",100)
    mu = kwargs.get('mu',0.4)
    phi = kwargs.get('phi',0.1)
    B = kwargs.get("rac",100)
    print_LP = kwargs.get("print_LP",False)
    IDtype = kwargs.get("IDtype","fullName")
    fluxcol = kwargs.get("fluxcol","fluxValue")
    def_infl = kwargs.get("infl_default",10)

    compartmenttag = kwargs.get("compartmenttag","_e0")

    if not isinstance(compartmenttag,dict):
        compartmenttag = dict([(mod,compartmenttag) for mod in models.keys()])

    # if isinstance(media,pd.DataFrame):
    #     for ky,mod in models.items():
    #         kargs = kwargs.copy()
    #         kargs["compartmenttag"] = compartmenttag[ky]
    #         set_media(mod,**kargs)

    model_parameters,metabolites = format_models(models,compartmenttag)
    # U = kwargs.get("uptake",np.ones(len(metabolites)))
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

    LP_result,mod_ordering = sx_gurobi(model_parameters,mu,U,phi,B,print_LP=print_LP)

    fluxes = {}
    biomass = {}
    for i,mod in enumerate(mod_ordering):
        S,_,_,ex_inds,_ = model_parameters[mod] 
        all_exinds = np.concatenate([ex_inds[0],ex_inds[1]])
        all_exinds = all_exinds[all_exinds != -1]         
        internal_indices = np.array([j for j in range(S.shape[1]) if j not in all_exinds])
        ##Exchange reported as Net flux OUT of the microbe
        exfl = np.array([LP_result.getVarByName("V_{}[{}]".format(mod,j)).x if j != -1 else 0 for j in ex_inds[0]]) - np.array([LP_result.getVarByName("V_{}[{}]".format(mod,j)).x if j != -1 else 0 for j in ex_inds[1]])
        flx_dict = dict([(metabolites[k],exfl[k]) for k in range(len(metabolites)) if ex_inds[0][k] != -1])
        intfl = dict([(j,LP_result.getVarByName("V_{}[{}]".format(mod,j)).x) for j in internal_indices])
        fluxes[mod] = {"Exchange":flx_dict,"Internal":intfl}
        biomass[mod] = LP_result.getVarByName("X[{}]".format(i)).x

    exprt = dict([(metabolites[i], LP_result.getVarByName("Export[{}]".format(i)).x) for i in range(len(U))])

    return {"Biomass": biomass,"Flux":fluxes,"CommunityExport":exprt}

def set_media(model,**kwargs):

    """
    Function to reconcile a media file from AGORA with a cobra model and attempt to set the model media with that file. **Unused** because media effects community uptake bounds rather than model by model bounds.

    :param model: a cobra model 
    :type model: GSM
    
    :param media: table including flux bound and modelSEED metabolite ID
    :type media:

    :param IDtype: How the MODEL ids the metabolites. Should be a column of the media table.
    :type IDtype: str

    :param compartmenttag: how the MODEl tags exchanged metabolites (e.g. _e0 for a modelSEED model)
    :type compartmenttag: str

    :param fluxcol: column in media table with flux bound
    :type fluxcol: str

    :param keep_fluxes: If True, the new media will include the fluxes from the models previous media that do not appear in media. Default False.
    :type keep_fluxes: bool

    :return: None

    """

    media = kwargs.get("media")

    IDtype = kwargs.get("IDtype","modelSeedID")
    compartmenttag = kwargs.get("compartmenttag","_e0")
    fluxcol = kwargs.get("fluxcol","fluxValue")
    keep_fluxes = kwargs.get("keep_fluxes",False)

    new_media = {}
    for rw in media.index:
        ### Check if the metabolite is exchanged by the model
        exchng_rxn = "EX_{}{}".format(media.loc[rw,IDtype],compartmenttag)
        # print(exchng_rxn)

        if exchng_rxn in model.reactions:
            #if it is, add it to the new media
            new_media[exchng_rxn] = media.loc[rw,fluxcol]

    if keep_fluxes:
        old_media = model.medium
        for ky in old_media.keys():
            if ky not in new_media.keys():
                new_media[ky] = old_media[ky]

    model.medium = new_media

    return None

def format_models(cobra_models,model_extag):

    """Creates tuples of stoichiometric matrix S, lower bounds, upper bounds, map from exchanged metabolites to exchange reaction index (export and uptake), and objective function.
    Also splits apart uptake and export exchange reactions. Does this for each model in a list of cobra models. Returns as dictionary keyed by model name.

    :param cobra_models: list or dict of cobrapy model objects
    :type cobra_models: list or dict

    :param model_extag: indicates how each model tags exchanged metabolites (e.g. _e0, _e)
    :type model_extag: list/dict

    :return: Set of models formatted for use in :py:func:`sx_gurobi <steadycomX.sx_gurobi>`
    :rtype: dict

    """

    from cobra import util

    if not isinstance(cobra_models,dict):
        modeldict = {}
        ordering = [mod.name for mod in cobra_models]
        for mod in cobra_models:
            modeldict[mod.name] = mod
        cobra_models = modeldict
    else:
        ordering = cobra_models.keys()
    
    if not isinstance(model_extag,dict):
        extagdict = {}
        for i,mod in enumerate(ordering):
            extagdict[mod] = model_extag[i]
        model_extag = extagdict

    exchanged_metabolites = {}
    exchange_reactions = {}
    exchanged_metabolites_ids = {}
    exchange_metabolites_nametoid = {}
    reaction_to_metabolite = {}
    metabolite_to_reaction = {}

    for modelkey in ordering:
        model = cobra_models[modelkey]

        #list all reactions the model claims are exchange.
        exchng_reactions =[rxn.id for rxn in  model.exchanges]#[rxn.id for rxn in model.reactions if 'EX_' in rxn.id]#

        exchng_metabolite_ids_wrx = [(rx,metab.id) for rx in exchng_reactions for metab in model.reactions.get_by_id(rx).reactants] #
        exrxn_to_met = dict(exchng_metabolite_ids_wrx)# a dictionary keyed by reaction reaction id with value exchanged metabolite id
        met_to_exrxn = dict([(t[1],t[0]) for t in exchng_metabolite_ids_wrx])# a dictionary keyed by metabolite ID with value exchange reaction
        exchng_metabolite_ids = [t[1] for t in exchng_metabolite_ids_wrx]# list of metabolite IDs for exchanged metabolites.


        exchng_metabolite_names = [model.metabolites.get_by_id(metab).name.replace(model_extag[modelkey],"") for metab in exchng_metabolite_ids] #list of exchanged metabolite names, with exchange tag removed


        idtonm = dict(zip(exchng_metabolite_ids,exchng_metabolite_names))
        nmtoid = dict(zip(exchng_metabolite_names,exchng_metabolite_ids))


        exchanged_metabolites[modelkey] = exchng_metabolite_names
        exchanged_metabolites_ids[modelkey] = exchng_metabolite_ids
        exchange_reactions[modelkey] = exchng_reactions
        exchange_metabolites_nametoid[modelkey] = nmtoid
        reaction_to_metabolite[modelkey] = exrxn_to_met
        metabolite_to_reaction[modelkey] = met_to_exrxn


    ##### NOW: we have to reconcile the exchanged metabolite and agree on an ordering
    masterlist = []
    for li in exchanged_metabolites.values():
        masterlist += li
    masterlist = np.unique(masterlist)

    ### Now we can form the tuples we need.

    formatted_models = {}

    for modelkey in ordering:
        model = cobra_models[modelkey]
        #Get the stoichiometric matrix
        ###Index is metabolite ID, columns are rxn ID
        S_df = util.array.create_stoichiometric_matrix(model, array_type = 'DataFrame')

        if len(exchanged_metabolites_ids[modelkey])  == len(exchange_reactions[modelkey]):
            S_df.loc[:,["REV{}".format(er) for er in exchange_reactions[modelkey]]] = -S_df.loc[:,exchange_reactions[modelkey]].values
            EyE = S_df.loc[np.array(exchanged_metabolites_ids[modelkey]),np.array(exchange_reactions[modelkey])]

            if np.all(EyE == -np.eye(EyE.shape[0])):#If -I, then negative flux of exchange reaction corresponds to metabolite uptake (this is the normal convention)
                export_indicies = [list(S_df.columns).index(metabolite_to_reaction[modelkey][exchange_metabolites_nametoid[modelkey][met]]) if met in exchanged_metabolites[modelkey] else -1 for met in masterlist]
                uptake_indicies = [list(S_df.columns).index("REV{}".format(metabolite_to_reaction[modelkey][exchange_metabolites_nametoid[modelkey][met]])) if met in exchanged_metabolites[modelkey] else -1 for met in masterlist]
            elif np.all(EyE == np.eye(EyE.shape[0])):#else then positive flux of exchange reaction corresponds to metabolite uptake
                uptake_indicies = [list(S_df.columns).index(metabolite_to_reaction[modelkey][exchange_metabolites_nametoid[modelkey][met]]) if met in exchanged_metabolites[modelkey] else -1 for met in masterlist]
                export_indicies = [list(S_df.columns).index("REV{}".format(metabolite_to_reaction[modelkey][exchange_metabolites_nametoid[modelkey][met]])) if met in exchanged_metabolites[modelkey] else -1 for met in masterlist]
            else:
                raise ValueError("[format_models] Error: Exchange reactions for {} not +/- identity".format(modelkey))

        else:
            raise ValueError("[format_models] Error: We do not support separated uptake/export exchange reactions.")

        lower_bounds = np.array([model.reactions.get_by_id(rx).lower_bound if rx[:3]!='REV' else model.reactions.get_by_id(rx[3:]).lower_bound for rx in S_df.columns])
        upper_bounds = np.array([model.reactions.get_by_id(rx).upper_bound if rx[:3]!='REV' else -model.reactions.get_by_id(rx[3:]).lower_bound for rx in S_df.columns])

        # lower_bounds = np.array([model.reactions.get_by_id(rx).lower_bound for rx in S_df.columns])
        # upper_bounds = np.array([model.reactions.get_by_id(rx).upper_bound for rx in S_df.columns])

        lower_bounds[export_indicies[export_indicies != -1]] = 0
        lower_bounds[uptake_indicies[uptake_indicies != -1]] = 0

        growth_col = pd.Series(np.zeros(S_df.shape[1]),index = S_df.columns)
        for rxn in util.solver.linear_reaction_coefficients(model).keys():
            growth_col.loc[rxn.id] = util.solver.linear_reaction_coefficients(model)[rxn]


        objective = growth_col.values

        # export_indicies = []
        # uptake_indicies = []

        formatted_models[modelkey] = (S_df.values,lower_bounds,upper_bounds,(export_indicies,uptake_indicies),objective)

    return formatted_models,masterlist

def sx_gurobi(model_params,mu,U,phi,RAC,print_LP = False):#,crossfed = None):

    """
    Function to form SteadyComX and solve as described in :cite:`kim2022resource` using Gurobi LP solver.

    :param model_params: tuples of stoichiometric matrix S, lower bounds, upper bounds, map from exchanged metabolites to exchange reaction index (see below), model objective
    :type model_params: list/dict

    :param mu: specific community growth rate
    :type mu: float

    :param U: Upper bound of community uptake of metabolites 
    :type U: array

    :param phi: forced metabolite leak
    :type phi: float

    :param RAC: Intracellular flux budget for RAC
    :type RAC: float 

    :param print_LP: Option to print the constraints of the linear program. default False
    :type print_LP: bool

    :return: Biomass X and metabolic flux V/X for each model. dict keys match model_params, or are index in model_params list. Exchange flux is given as net flux OUT of the microbe (i.e. effect on environmental pool)
    :rtype: dict

    The map from exchanged metabolites to exchange reaction should be a tuple with (export,uptake). export should be a list such that export[i] is the index of the exchange reaction that exchanges metabolite[i], or -1 if that 
    metabolite is not exchanged by the model.



    """


    if isinstance(model_params,dict):
        mod_ordering = list(model_params.keys()) 
    else:
        mod_ordering = list(range(len(model_params)))
        model_params = dict([(i,model_params[i]) for i in mod_ordering])

    sx_LP = gb.Model("sx_LP")
    sx_LP.setParam( 'OutputFlag', False )

    #(Eq 5 - X>=0)
    X = sx_LP.addMVar(len(mod_ordering),lb=0,name = 'X',ub = gb.GRB.INFINITY)
    Uv = sx_LP.addMVar(len(U),lb=0,name = 'Uptake',ub=U)
    sx_LP.update()
    Ev = sx_LP.addMVar(len(U),lb=0,name = 'Export',ub = gb.GRB.INFINITY)
    sx_LP.update()
    #(Eq 1 - Maximize total biomass)
    sx_LP.setMObjective(None,np.ones(len(mod_ordering)),0,xc = X, sense= gb.GRB.MAXIMIZE)
    sx_LP.update()


    for i,mod in enumerate(mod_ordering):
        S,lb,ub,ex_inds,objective = model_params[mod] 
        V = sx_LP.addMVar(S.shape[1],name = 'V_{}'.format(mod),lb=-gb.GRB.INFINITY, ub = gb.GRB.INFINITY)
        sx_LP.update()

        x = sx_LP.getVarByName("X[{}]".format(i))

        # #(Eq 2 - Chemical Equilibrium)
        sx_LP.addMConstr(S,V,"=",np.zeros(S.shape[0]),name="Eq2_{}".format(mod))
        sx_LP.update()


        # sx_LP.addMConstr(-np.identity(S.shape[1]),V,"<=",-lb,name="Eq3Lower_{}".format(mod))
        # sx_LP.update()
        # #(Eq 3 - upper bounds)
        # sx_LP.addMConstr(np.identity(S.shape[1]),V,"<=",ub,name="Eq3Upper_{}".format(mod))
        # sx_LP.update()
    

        #(Eq 3 - Upper/Lower bounds (Do I need to adjust something here for crossfeeding? Mass balance should take care of that...)
        vars = np.concatenate([[x],np.array(V.tolist())])
        #(Eq 3 - lower bounds)
        lbC = np.concatenate([np.array([lb]).T,-np.identity(len(lb))], axis = 1)
        sx_LP.addMConstr(lbC,gb.MVar(vars),"<=",np.zeros(len(lb)),name="Eq3Lower_{}".format(mod))
        sx_LP.update()
        #(Eq 3 - upper bounds)
        ubC = np.concatenate([-np.array([ub]).T,np.identity(len(lb))], axis = 1)
        sx_LP.addMConstr(ubC,gb.MVar(vars),"<=",np.zeros(len(lb)),name="Eq3Upper_{}".format(mod))
        sx_LP.update()
        
        #(Eq 4 - growth)
        scld_objective = mu*np.array(objective)
        sx_LP.addConstr(x == scld_objective @ V.tolist(),name="Eq4_{}".format(mod))
        sx_LP.update()

        #(Eq 8 - Resource Allocation)
        # To get absolute values, we need auxilary variables.
        all_exinds = np.concatenate([ex_inds[0],ex_inds[1]])
        all_exinds = all_exinds[all_exinds != -1]       
        internal_indices = np.array([j for j in range(S.shape[1]) if j not in all_exinds])
        absV = sx_LP.addMVar(len(internal_indices),name="ABSV_{}".format(mod),lb=0,ub = gb.GRB.INFINITY)#np.empty(len(internal_indices),dtype=gb.Var)
        for k,inti in enumerate(internal_indices):
            sx_LP.addGenConstrAbs(absV[k],V[inti])
        
        sx_LP.update()
        sx_LP.addConstr(np.ones_like(absV.tolist()) @ np.array(absV.tolist()) <= RAC*x,name="Eq8[{}]".format(mod))
        
        sx_LP.update()

    for i in range(len(U)):
        export_list = np.array([sx_LP.getVarByName("V_{}[{}]".format(mod,model_params[mod][3][0][i])) for mod in mod_ordering if model_params[mod][3][0][i] != -1])
        uptake_list = np.array([sx_LP.getVarByName("V_{}[{}]".format(mod,model_params[mod][3][1][i])) for mod in mod_ordering if model_params[mod][3][1][i] != -1])

        # (Eq 6 - Community Balance)
        sx_LP.addConstr(Uv[i] - Ev[i] + np.ones(len(export_list)) @ export_list - np.ones(len(uptake_list)) @ uptake_list == 0,name="Eq6[{}]".format(i))
        sx_LP.update()

        # (Eq 7 - Forced Leak)
        if len(uptake_list):
            sx_LP.addConstr(phi*np.ones(len(export_list)) @ export_list <= Ev[i],name="Eq7[{}]".format(i))
            sx_LP.update()

    if print_LP:
        for c in sx_LP.getConstrs():
            exp = "{}".format(sx_LP.getRow(c))
            print("{}: {} {} {}".format(c.ConstrName,exp.split(": ")[-1].replace(">",""),c.Sense,c.RHS))


    sx_LP.optimize()
    sx_LP.update()

    return sx_LP,mod_ordering

def steadyComXLite(models,**kwargs):

    """

    Computes steadyComX simulation, returning only biomasses
    
    :param models: a dictionary of models formatted by :py:func:`format_models<steadycomX.format_models>`
    :type model: dict[GSM]

    :param mu: specific community growth rate
    :type mu: float

    :param uptake: Upper bound of community uptake of metabolites 
    :type uptake: array

    :param phi: forced metabolite leak
    :type phi: float

    :param rac: Intracellular flux budget for RAC
    :type rac: float 

    :param print_LP: Option to print the constraints of the linear program. default False
    :type print_LP: bool

    :return: biomass dictionary, keyed by model name
    :rytpe: dict

    """
    mu = kwargs.get('mu',0.4)
    phi = kwargs.get('phi',0.1)
    B = kwargs.get("rac",100)
    print_LP = kwargs.get("print_LP",False)

    U = kwargs.get("uptake")

    LP_result,mod_ordering = sx_gurobi(models,mu,U,phi,B,print_LP=print_LP)

    biomass = {}
    for i,mod in enumerate(mod_ordering):
        biomass[mod] = LP_result.getVarByName("X[{}]".format(i)).x

    return biomass