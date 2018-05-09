import numpy as np 
import re
import os


# vmeasure is harmonic mean of distinct homogeneity and completeness
# assume classes C and clusters K
# in our case, classes are true segmentations and clusters are discovered segmentations
# A = {a_i,j} where a_i,j is number of datapoints in class c_i that were assigned to cluster k_j
# homogeneity: if all clusters only contain datapoints from a single class
# completeness: if all datapoints that are in the same class are in the same cluster
# if only one class, homogeneity is 1

#	  | 1 if H(C,K) = 0
# h = |
#     | 1- H(C|K)/H(C) else

# H(C|K) = - sum_over_clusters sum_over_classes a_i,j/N * log(a_i,j/ sum_over_classes a_c,j)
# H(C) = - sum_over_classes (sum_over_clusters a_i,j)/n * log(sum_over_clusters a_i,j)/ n)

# where N is number of datapoints and n is number of classes

# A format
# 		clusters
# classes   1,2,3,4,5...
# 		  1 
# 		  2
# 		  3
# 		  4
# 		  5


def homogeneity(classes, clusters, A):
	if len(classes) == 1:
		return 1
	cond_e = cond_entropy(classes, clusters, A)
	e = entropy(classes, clusters, A)

def cond_entropy(classes, clusters, A):
	N = np.sum(np.sum(A, axis=1), axis=0)
	print(N)
	right = A/N
	print(np.sum(A, axis=1))
	divisor = np.repeat(np.sum(A, axis=1).reshape(-1,1), A.shape[0], axis=1)
	print(divisor)
	print(np.divide(A, divisor, where=divisor!=0))
	left = np.log(np.divide(A, divisor, where=divisor!=0))/len(classes)
	return -np.sum(np.sum(np.multiply(right,left), axis=1), axis=0)

def entropy(classes, clusters, A):
	pass


if __name__ == '__main__':
	test_classes = np.arange(0,3,1)
	test_clusters = np.arange(0,3,1)
	test_A = np.array([[5,3,1],
					   [1,5,2],
					   [0,2,6]])
	test_B = np.eye(3)

	print(cond_entropy(test_classes, test_clusters, test_A))
