import numpy as np
from numpy import *
import pandas as pd
from scipy.sparse.linalg import eigsh
import scipy.sparse.linalg as lg
import time

Z= pd.read_csv('filename.csv',index_col=0,header=0)
Z = Z.select_dtypes(include='number')
Y = np.array(list(Z.values))
def cal_pairwise_dist(x):
global sum_x,dist
'''Calculation of Pairwise Distance, x is a matrix.
Returns the square of the distance between any two points
(a-b)^2 = a^2 + b^2 - 2*a*b
'''
sum_x = np.sum(np.square(x), 1)
dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
return dist
start_cal_pair_time=time.time()
A=cal_pairwise_dist(Y)
end_cal_pair_time = time.time()
elapsed_time_cal_pair = end_cal_pair_time - start_cal_pair_time
print(elapsed_time_cal_pair)
start_knn_time=time.time()
k = 5
'''number of nearest neighbors for the graph.'''
DNN, NN = np.sort(A), np.argsort(A)
'''get indices and sorted distances
np.argsort()'s output will display the indices of a sorting element'''
NN = NN[:,1:k+1]
DNN = DNN[:,1:k+1]
edges = []
for i in range(len(NN)):
for j in range(len(NN[i])):
edges.append((i+1, NN[i][j] + 1))
end_knn_time=time.time()
elapsed_time_knn = end_knn_time - start_knn_time
print(elapsed_time_knn)
def heat_kernel(x, y):
global w, dist_2
'''heat kernel for edge weights'''
dist_2 = np.linalg.norm(x - y)**2
w = np.exp(-dist_2)
return w
def my_LE(X,k):
global Y, W, D, L
'''my_LE(data, k = integer)
k: neighbors
Example: my_LE(dataName,5)
'''
W = np.zeros((Y.shape[0], Y.shape[0]))
'''Creation of adjacency matrix'''
for i in range(W.shape[0]):
for j in range(W.shape[0]):
if j in NN[i]:
W[i][j] = heat_kernel(Y[i], Y[j])

'''The distance of two nodes. If distance is close edge weight will

be higher.'''
node_degrees = W.sum(axis=1)
D = np.diag(node_degrees)
'''Creation diagonal degree matrix'''
L = D - W
'''Formula of laplacian matrix'''
return L
start_LE_time=time.time()
M = my_LE(Y,5)
end_LE_time = time.time()
elapsed_LE = end_LE_time - start_LE_time
print(elapsed_LE)
# In[ ]:
start_eigs_time=time.time()
global eigvals, eigvecs
eigvals, eigvecs=eigsh(L,k=10,which='LM')
'''
k:The number of eigenvalues and eigenvectors desired. k must be smaller than N.
It is not possible to compute all eigenvectors of a matrix.
eigs:Find k eigenvalues and eigenvectors of the square matrix A
eigsh:Find k eigenvalues and eigenvectors of the real symmetric square matrix or
complex Hermitian matrix A'''

eigvals
eigvecs
print("")
print("Largest eigenvalue:", max(eigvals))
eigvecs = pd.DataFrame(eigvecs)
end_eigs_time = time.time()
elapsed_eigs = end_eigs_time - start_eigs_time
print(elapsed_eigs)
fm,fn=shape(eigvecs)
print ('fm,fn:',fm,fn)
lamdaIndicies = argsort(eigvals)
'''Sorting the eigenvalues low to high'''
first=0
second=0
print (lamdaIndicies[0], lamdaIndicies[-1])
'''printing of the two smallest eigenvalues'''
for i in range(fm):
'''dimension'''
if eigvals[lamdaIndicies[i]].real>1e-5:
'''.real function return the real part of the complex argument'''
print ("lamda:",eigvals[lamdaIndicies[i]],"lamda index:",lamdaIndicies[i])
first = lamdaIndicies[i]
second = lamdaIndicies[i+1]
'''the index of the smallest two eigenvalue'''
break
print (first, second)
eigvals
eigvecs=eigvecs.values
redEigVects = eigvecs[:,lamdaIndicies]
eigvecs
start_argsort_time=time.time()
lamdaIndicies = argsort(eigvals)
end_argsort_time=time.time()
elapsed_time_argsort=end_argsort_time - start_argsort_time
print(elapsed_time_argsort)
output_measurements =
{'Operation':['cal_pair_time','knn_calculation_time','le_time','eigen_sorting_time'
,'eigen_calculation_time'],
'Observation':
[elapsed_time_cal_pair,elapsed_time_knn,elapsed_LE,elapsed_time_argsort,elapsed_eig
s] }
output_df = pd.DataFrame(output_measurements)
print(output_df)
output_df.to_csv(r'LE_Measure.csv')
