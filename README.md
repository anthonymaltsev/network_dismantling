# Greedy Approaches utilizing Centrality for Node-Weighted Network Dismantling
## Anthony Maltsev, Rishi Nath, Alejandro Sanchez Ocegueda

This repository contains code and results from our research project for CS270 on generalized network dismantling.

Our study is centered around the use of centrality to develop greedy algorithms for the network dismantling problem, which is NP-complete. In graph theory, network dismantling refers to the problem of identifying a set of vertices whose removal would break the graph into connected components of at most a given target size. In our study, we do a survey of existing network dismantling algorithms and heuristics for node-weighted graphs. We also suggest new approaches to modifying unit-case centrality heuristics to the node-weighted case. Finally, we implement many of our surveyed and suggested approaches to compare their performance on a diverse set of node-weighted graphs. We also introduce a new scoring method for generalized network dismantling algorithms called weighted robustness.

## Code

Code to benchmark each different centrality measure that we studied and the GND algorithm can be found in the `dismantling_code` folder.

The code relies on the following libraries:
 * numpy
 * networkx
 * scipy

The `Centrality_Benchmarking.ipynb` notebook contains code to benchmark different algorithms. To run each benchmarking function, first define a list of graphs on which you want to benchmark, and define a heuristics dictionary which specifies which centrality measures you want to benchmark.

The notebook `Experimentation.ipynb` is with some starter code to explore and visualize different dismantling algorithms. 

Results for each algorithm and graph type and size can be found in the 3 different results csvs.

## Acknowledgements

Thank you to Jelani Nelson, our professor, who acted as an advisor for our project.
