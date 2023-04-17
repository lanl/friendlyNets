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

Using the Method for a Set of Samples
----------------------------------------

.. note::

   What follows is a plan of action. At this point, the data formatting and network building steps are incomplete.

FriendlyNets is designed to predict the invasion of a single species into a community. It is designed around microbiome studies, but (hopefully) written more generally so that nodes can be anything as long as interaction parameters between
the nodes are known. For microbiome studies, we provide a method to generate the set of interaction parameters from a set of genome-scale metabolic models which must be provided by the user.

We assume that the user has a table in .csv format with rows indexed by species of interest and columns indexed by sample name that contains abundance data for each species in each sample (relative or absolute). Additionally, if an assessment of the predictive power of the method is desired (which may indicate
the extent to which experimental outcome depends network effects), a metadata file must be provided as .csv with rows indexed by sample name and a column indicating experimental outcome of an invasion experiment

Creating the full network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Not Implemented

.. note::

    Optional: A network can instead be loaded as pandas DataFrame with nodes and columns being names of the species in the data, formatted  as ``Network.loc[m_1,m_2]`` contains the interaction with source ``m_1`` and target ``m_2``.

For microbiome data paired with a set of genome-scale metabolic models, the method creates network of interactions for all co-occurring taxa that have an associated user-provided genome-scale model. 

.. code-block:: python

    from make_gem_network import make_gem_network

    full_interaction_network = make_gem_network(path_to_table,path_to_gsm_info) 

This function requires the path to the abundance table file (a .csv) and the path to a file containing the paths to each genome-scale model.

By default, the interaction will be defined by the log-ratio of simulated growth in the pair to simulated growth alone. 

Formatting the data
^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Not Implemented

.. warning::

    Removes any species from the data that do not have interaction parameters in the full interaction network.

We need to format the samples into a dict of dicts, or dict of tuples if predicitive power is to be assessed. 

To format the data to compute friendliness scores with no known outcomes:

.. code-block:: python

    from format_data import format_data

    experiment,scoretype = format_data(path_to_table,full_interaction_network,nodes_of_interest)

The function requires the path to the table of abundances and the full interaction network so that species missing can be removed, and the name of the node(s) that we wish to assess for network friendliness. 

Without known outcomes, ``experiment`` will be a dict (keyed by sample name - the column headers of the abundance table)
of dicts (keyed by species), and ``scoretype`` will be None

For assessment of the predictive power of the method, the path to a metadata file with known outcomes for each sample is required, as is the name of the known outcome column in that file (default ``Score``):

.. code-block:: python

    from format_data import format_data

    experiment,scoretype = format_data(path_to_table,full_interaction_network,nodes_of_interest,known_scores = path_to_metadata,score_column = column_of_score,scoretype = score_type)



If the known outcome file is given, along with the name of the column of known outcome scores, ``experiment`` will be a dict (keyed by sample name - the column headers of the abundance table) of 
tuples with (known outcome score, dict of abundances). The dict of abundances is keyed by species. In this case, the function attempts to guess if the known scores are binary or continuous unless the
scoretype is given. If the scoretype is given as binary and the data are continuous, the function binarizes the data.

The second return value, ``scoretype`` indicates the type of known outcome scores, either binary or continuous.

Computing Friendliness Scores and Assessing Predictive Power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compute friendliness scores for each sample on a node(s) of interest

.. code-block:: python

    from score_net import network_friendliness

    friendliness = {}
    for target_node in nodes_of_interest:
        friendliness[target_node] = network_friendliness(experiment,full_interaction_network,target_node)

The return value is a pandas dataframe that can be saved as a .csv file.

To assess the predictive power of the method (for friendliness on a single ``target_node`` in ``nodes_of_interest``)

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