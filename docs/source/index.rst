.. FriendlyNets documentation master file, created by
   sphinx-quickstart on Thu Mar 16 08:36:48 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FriendlyNets documentation
========================================

FriendlyNets provides a method for assessing the promotion/inhibition effect on a microbe of a microbial community using a network of community interactions. At its core, FriendlyNets judges how much a network
promotes or inhibits one of its nodes. It does this by assuming a set dynamical system represented by the network, and using the resulting dynamics. FriendlyNets is also packaged with functions for generating 
a network from a set of genome-scale metabolic by simulating pairwise growth using the methods from :cite:`kim2022resource`.



Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project. 

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   format
   network_build
   scoring
   sensit
   friendlynets


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. bibliography:: reference.bib