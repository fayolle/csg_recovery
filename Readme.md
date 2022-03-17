# Point-cloud to CSG 
Simple implementation of the genetic programming code used in this [paper](https://doi.org/10.1016/j.cad.2016.01.001) to generate a CSG expression from a 3D point-cloud. 


## How to compile
Compile the segmentation program (based on Ransac) in the subdirectory 'segmentation' and the CSG tree search program (based on genetic programming) in the subdirectory 'gp'. 
Both programs rely on the library boost. 

The code in the subdirectory 'segmentation' is a modified version of the code from the efficient RANSAC [paper](https://doi.org/10.1111/j.1467-8659.2007.01016.x).


## How to run
Assuming that the compiled programs for the segmentation and the CSG tree search are copied in the subdirectory 'bin'. 

### Step 1: Segmentation
```
bin/ransac_command data/test.xyzn out/test-segmented.segps [data/test.conf] > out/log.txt
```
The program 'ransac_command' has three arguments: the name of the input 3D point-cloud (first argument), the name of the output point-cloud with segmentation information (second argument), a config file (optional). 

### Step 2: Recovery step
Produce an expression for the CSG tree
```
bin/gp out/test-segmented.fit data/test.xyzn out/test_expression.txt out/test_primitives.txt 1 150 1000 0.3 0.4 > out/gplog.txt
```
The program 'gp' has the following arguments:
* The name of the file with the list of primitives (computed by 'ransac_command')
* The name of the input 3D point-cloud 
* The name of the file where the CSG expression will be written 
* The name of the file where the list of primitives used in the CSG expression will be written 
* The level of information to be displayed (0: nothing, >0: print information) 
* The number of creatures in the population 
* The number of iterations 
* The mutation rate 
* The crossover rate 

### Step 3: C source file generation and evaluation
```
python utils/create_eval_source.py out/test-segmented.fit out/test_expression.txt out/test.cpp data/test.xyzn 
```
The C++ source code with the CSG expression can be found in the file 'out/test.cpp'. The function defined in the C++ file is an implicit surface (approximate SDF) that can be rendered by ray-tracing or after meshing with the Marching Cubes algorithm. 

Note: It is possible to generate a tree (using Graphviz format) corresponding to the CSG expression. 
```
python utils/tree_from_expression.py out/test_expression.txt out/test_primitives.txt out/test_tree.dot 
```
The output file is in 'out/test_tree.dot'. It can be processed with the 'dot' command of GraphViz. 


## Additional notes
Obviously, the results depend a lot on the list of primitives passed to the CSG tree search. The version provided here uses [RANSAC](https://doi.org/10.1111/j.1467-8659.2007.01016.x). There are several ways to improve it such as, for example: Section 4.1 of this [paper](https://doi.org/10.1145/3272127.3275006), this [paper](https://doi.org/10.5220/0008870600380048) or this [one](https://doi.org/10.5220/0010297100750084). 


## Reference 
Link to the [paper](https://doi.org/10.1016/j.cad.2016.01.001) where the approach is described and the bibtex entry
```
@article{pc2csg2016,
title = {An evolutionary approach to the extraction of object construction trees from 3D point clouds},
journal = {Computer-Aided Design},
volume = {74},
pages = {1-17},
year = {2016},
issn = {0010-4485},
author = {Pierre-Alain Fayolle and Alexander Pasko},
}
```
