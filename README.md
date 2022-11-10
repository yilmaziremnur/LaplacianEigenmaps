# LaplacianEigenmaps
Laplacian Eigenmaps Python code without sklearn from dimension reduction methods.

The Laplacian Eigenmaps method consists of 3 basic steps. These steps are:
1- Creation of the neighborhood graph: If xi and xj are close, between nodes i and j
an edge is set. Two variations are available for this step.
a) ε-neighborhoods (ε ∈ R): Nodes i and j are connected by an edge if ∥xi − xj∥2 < ε.
b) n nearest neighbors (n ∈ N): Nodes I and j, nearest neighbors of j
or if j is among the nearest neighbors of i, it is joined by an edge.
2-Determining the weights: There are two variations for this step.
a) Heat Kernel:
b) Simple Minded: Set Wij = 1 if nodes i and j are connected, and Wij = 0 if they are not.
gets.
3-Eigenmaps (Lf = λDf): Eigenvalues ​​for the generalized eigenvector problem and
eigenvectors are calculated.
Here D is the diagonal weight matrix and its inputs are columns. (Since W is symmetrical, the line
It may also be) Dii = Σj Wji formed by the W sums.
Laplacian, on the other hand, is a symmetric, positive semi-definite matrix. Laplacian matrix with L = D-W
is formulated.
