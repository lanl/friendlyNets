# FriendlyNets

FriendlyNets provides a method for assessing the promotion/inhibition effect on a microbe of a microbial community using a network of community interactions. At its core, FriendlyNets judges how much a network
promotes or inhibits one of its nodes. It does this by assuming a set dynamical system represented by the network, and using the resulting dynamics. FriendlyNets is also packaged with functions for generating 
a network from a set of genome-scale metabolic by simulating pairwise growth using the methods from Kim et al., 2022[[1]](#1).

## Dependencies

We use `joblib` for parallel computing.

SteadyComX uses Gurobi for joint FBA, and takes COBRApy model objects as input.

## References
<a id="1">[1]</a> 
Kim, Minsuk, Jaeyun Sung, and Nicholas Chia. "Resource-allocation constraint governs structure and function of microbial communities in metabolic modeling." Metabolic Engineering 70 (2022): 12-22.

## Copywrite 

C# O4648

© 2023. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are.
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.
others to do so.
