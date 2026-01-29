# BIDijkstra

This repository provides an empirical evaluation for the shortest st-path problem in graphs with positive edge weights. 

The structure is as follows:

- Algorithms: Impelemntation of 3 algorithms, namely Dijkstra's Algorithm, Bidirectional Dijkstra's Algorithm as specified in [Bidirectional Dijkstra’s Algorithm is Instance-Optimal](https://doi.org/10.1137/1.9781611978315.16), and a shortest path algorithm tailored to a specific class of graphs.
- Graph_Generation: Functions for generating input graphs.
- Experiments: Functions for comparing runtime and number of relaxations of different approaches.
- Evaluation notebook: Runs the experiments and provides some visualizations.
