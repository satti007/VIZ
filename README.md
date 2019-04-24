## Algorithm Visualizer v2.0

### Instructions to run:
cd src/ <br />
python3 Alviz2Tree.py <br />

### Options in visualizer supported:
1. Number of leaf_nodes: The number of leaf nodes to be there in the generated tree
2. Branching Factor    : The number of max children for each node
3. lr : if it's 1, alpha-beta pruning runs from left to right, o/w right to left  

#### In data folder the following data files are stored in pickle format
1. nodes -- coords of the tree nodes -- as a list
2. edges -- edges connecting the tree nodes -- as a list of tuples where each tuple has two nodes 
3. node_values -- values for nodes (randomly generated) -- as a dict, i.e dict[node.id] = node_value
4. path -- the path traversed by the algorithm -- as a list
5. best -- the best path i.e (one of) the winning strategy for MAX player 
