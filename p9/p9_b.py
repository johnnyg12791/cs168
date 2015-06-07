import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import csv
from sets import Set
from collections import Counter
import sys



def main():
  input_file = "cs168mp9.csv"

  friendships = {}
  data = np.genfromtxt(input_file, dtype='int', delimiter=',')
  for row in data:
    f1, f2 = row[0], row[1]
    if f1 not in friendships:
      s = Set()
      friendships[f1] = s
    friendships[f1].add(f2)
    if f2 not in friendships:
      s = Set()
      friendships[f2] = s
    friendships[f2].add(f1)

  n = len(friendships)
  A = np.zeros((n, n))
  for i in friendships:
    for j in friendships[i]:
      A[i-1][j-1] = 1
  #print A.shape
  #print A

  
  d_array = np.zeros(n)
  for i in range(n):
    d_array[i] = np.sum(A[i])
  D = np.diag(d_array)

  L = D - A
  w, v = la.eig(L)

  sorted_indices = np.argsort(w)

    
  if sys.argv[1] == "graph" : 
    plt.scatter(v[:, sorted_indices[6]], v[:, sorted_indices[11]])
    plt.show()


  center_cluster = []
  left_cluster = []
  right_cluster = []

  graphed_vals = [(x,y) for x,y in zip(v[:,sorted_indices[6]], v[:,sorted_indices[7]])]

  index = 0
  for x, y in graphed_vals:
    if abs(x + .04) < .008 and abs(y - .04) < .008:
      center_cluster.append(index)
    index += 1

  index = 0

  for x,y in [(x,y) for x,y in zip(v[:,sorted_indices[6]], v[:,sorted_indices[11]])]:
    if x.real < -.04 and index not in center_cluster:
      left_cluster.append(index)
    index += 1

  index = 0

  #for x,y in [(x,y) for x,y in zip(v[:,sorted_indices[7]], v[:,sorted_indices[9]])]:
  #  if y.real > .012 and index not in center_cluster and index not in left_cluster:
  #    right_cluster.append(index)
  #  index += 1

  for i in range(1495):
    if i not in left_cluster and i not in center_cluster:
      right_cluster.append(i)

  print "The left cluster has ", len(left_cluster), " points"
  print "The center cluster has ", len(center_cluster), " points"
  print "The right cluster has ", len(right_cluster), " points"




  '''
  w = np.around(w, 10)
  num_connected = 0
  for i in range(1, w.shape[0] - 1):
    if w[i] == 0:
      num_connected += 1
  print num_connected
  '''

  #We want to find the clusters
  #random_set = np.random.choice(1495, size=350, replace=False)
  print "Left, Center, Right"
  for random_set in [left_cluster, center_cluster, right_cluster]:

    S = {}
    not_S = {}
    for i in range(len(friendships)):
      if i in random_set:
        S[i+1] = friendships[i+1]
      else:
        not_S[i+1] = friendships[i+1]


    print conductance(A, S, not_S)


#S is a set of points (7, 199, 2413...)
#A is the adjacency matrix, a set of tuples  
def conductance(A, S, not_S):

  numerator = 0.0
  for i in S.keys():
    for j in not_S.keys():
      numerator += A[i-1, j-1]

  denominator = min(adjacency(A, S), adjacency(A, not_S))

  #print numerator, denominator

  return numerator/denominator

#S is a dictionary, A is a matrix
#Need to compute the number of elements 
def adjacency(A, S):
  adj = 0
  edges_used = []
  for u in S.keys():
    for v in S[u]:
      if (u, v) not in edges_used:
        #1 if adjacent, 0 if not
        adj += A[u-1, v-1]
        edges_used.append((u,v))
  return adj


if __name__ == "__main__":
  main()




