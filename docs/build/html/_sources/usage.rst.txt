Usage
=====

.. _installation:

Installation
-------------

To use FriendlyNets, clone from github:

.. code-block:: console

    $ git clone https://github.com/jdbrunner/friendlyNets.git

(We plan to add pip installation in the future)

You will also need to add the directory to your python path, for example using

.. code-block:: python

    import sys
    import os
    sys.path.append(os.path.join(os.path.expanduser("~"),location,"friendlyNets"))

where ``location`` is the path to the folder that you cloned friendlyNets into.

**Dependencies**

We use `joblib <https://joblib.readthedocs.io/en/latest/>`_ for parallel computing.

Our implementation of SteadyComX uses `Gurobi <https://www.gurobi.com/documentation/9.5/>`_ and gurobi's python package for joint FBA, and expects `cobrapy <https://opencobra.github.io/cobrapy/>`_ models as input.

.. note::

    Unfortunately, we do not currently have access to a CPLEX license in order to implement CPLEX as a solver option.

Using the Method for a Set of Samples
----------------------------------------


FriendlyNets is designed to predict the invasion of a single species into a community. It is designed around microbiome studies, but written more generally so that nodes can be anything as long as interaction parameters between
the nodes are known. For microbiome studies, we provide a method to generate the set of interaction parameters from a set of genome-scale metabolic models which must be provided by the user.

We assume that the user has a table in .csv format with rows indexed by species of interest and columns indexed by sample name that contains abundance data for each species in each sample (relative or absolute). Additionally, if an assessment of the predictive power of the method is desired (which may indicate
the extent to which experimental outcome depends on network effects), a metadata file must be provided as .csv with rows indexed by sample name and a column indicating experimental outcome of an invasion experiment

Formatting the data
^^^^^^^^^^^^^^^^^^^^^^

We need to format the samples into a specific (not very user-friendly) form for use with the method. We provide :doc:`format` so that the user can provide data in an easier way.

.. note::

    Often, a significant amount of pre-processing is required to arrange the data into the required format for our formatting function (e.g. splitting into seperate experiments, etc.). We have provided three examples in the in 
    ``Example`` folder.

To format the data to compute friendliness scores with no known outcomes:

.. code-block:: python

    from format_data import format_data

    experiment,scoretype,coverage = format_data(otu_dataframe)

The function requires the abundance data table (as pandas dataframe) and returns a set of samples formatted for use in downstream analysis. It also returns two variables that are not relevant if called as above. 

Without known outcomes, ``experiment`` will be a dict (keyed by sample name - the column headers of the abundance table)
of dicts (keyed by species), and ``scoretype`` will be None. Coverage will be a dictionary simply indicating that the entire index of the abundance data table is included.

For assessment of the predictive power of the method, the path to a metadata file with known outcomes for each sample is required, as is the name of the known outcome column in that file (default ``Score``). Furthermore, 
the function can filter the data, including only a subset of the index of the abundance table. This can be used to filter out taxa for which no genome scale model is available.

.. code-block:: python

    from format_data import format_data

    experiment,scoretype,coverage = format_data(otu_dataframe,sample_metadata = metadata_dataframe,score_col = "Score",included_otus = taxa_with_models)


If the known outcome file is given, along with the name of the column of known outcome scores, ``experiment`` will be a dict (keyed by sample name - the column headers of the abundance table) of 
tuples with (known outcome score, dict of abundances). The dict of abundances is keyed by species. In this case, the function attempts to guess if the known scores are binary or continuous unless the
scoretype is given. If the scoretype is given as binary and the data are continuous, the function binarizes the data.

The second return value, ``scoretype`` indicates the type of known outcome scores, either binary or continuous.

The third return value, ``coverage`` is a dict of dicts indicating the coverage of the samples by the included otus. For each sample, the corresponding dictionary has the keys:

 - *Coverage* : The total relative abundance of the otus included.
 - *NumberMissing* : The number of otus not included.
 - *MajorMissing* : The otus not included with the highest relative abundance (any otus with :math:`\geq 80%` of the highest missing relative abundance)
 - *AllMissing* : List of all the otus not included.


Creating the full network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Optional: A network can instead be loaded as a pandas DataFrame with nodes and columns being names of the species in the data, formatted  
    as ``Network.loc[m_1,m_2]`` contains the interaction with source ``m_1`` and target ``m_2``.

.. note::

    This guide assumes that the user has GSMs corresponding to the organisms in their samples. In the ``Examples``, we provide examples of how one might construct these models (``make_models.py``) using `modelSEED <https://modelseed.org/>`_.

For microbiome data paired with a set of genome-scale metabolic models, the method creates network of interactions for all co-occurring taxa that have an associated user-provided genome-scale model. 

.. code-block:: python

    from make_gem_network import make_gem_network

    pair_growth,fba_growth,metadata = get_pairwise_growth(model_list,media_pth)

This function requires the path to a file containing the paths to each genome-scale model, as well as the path to a media file. Media files
can be found in the ``translate_agora_media`` directory, and updated by running the script ``get_agora_media.py``. Media needs to have a column containing
metabolite IDs matching those found in the GSMs, and a column containing the flux bounds for the media (both column names can be set by the user).

See :py:func:`make_gem_network.get_pairwise_growth` for a full list of the options available. If the experiments as created above
are passed, then the function skips pairwise growth experiments for any pair that does not co-occur in the dataset.

The result is a set of simulated pairwise growth experiment results, with ``pair_growth[i,j]`` being the growth of ``i`` when paired with ``j``. To create an interaction network, we can use these results in a few ways.
The simplest is to use the log-ratio of growth alone with growth in the pair as the Lotka-Volterra parameter for the partners influence on growth.



Computing Friendliness Scores and Assessing Predictive Power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compute friendliness scores for each sample on a node(s) of interest use :py:func:`score_net.network_friendliness`

.. code-block:: python

    from score_net import network_friendliness

    friendliness = {}
    for target_node in nodes_of_interest:
        friendliness[target_node] = network_friendliness(experiment,full_interaction_network,target_node)

The return value is a pandas dataframe that can be saved as a .csv file.

To assess the predictive power of the method (for friendliness on a single ``target_node`` in ``nodes_of_interest``) use py:func:`score_net.score_net`

.. code-block:: python

    from score_net import score_net

    friendliness,predictive_power = score_net(experiment,full_interaction_network,target_node,scoretype)

``predictive_power`` is a dictionary of predictive power metrics, which depend on if the scoring is binary (in which case the ROC is used) or continuous (in which case correlation is used). 



Plotting the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    I will probably add some functions to make plotting the results convenient. 

Sensitivity to Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide functionality to assess the sensitivity of the predictions to two types of perturbations. 

The first is sensitivity to community composition, which we test using simulated knock-outs (i.e. computing friendliness scores with nodes removed).

The second is sensitivity to the interaction parameter values. We test this using a dynamical system for :math:`\frac{\partial x_i}{\partial a_{ij}}`. See :doc:`sensit`.