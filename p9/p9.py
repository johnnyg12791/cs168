#p9.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    if(sys.argv[1] == "1"):
        part1()
    else:
        part2()


def part2():
    build_adjacency_matrix("cs168mp9.csv")



def build_adjacency_matrix(filename):
    f = open(filename, 'r')



def part1():

    if(sys.argv[2] == "a"):
        L = build_a_laplacian()
    if(sys.argv[2] == "b"):
        L = build_b_laplacian()
    if(sys.argv[2] == "c"):
        L = build_c_laplacian()
    if(sys.argv[2] == "d"):
        L =build_d_laplacian()

    #Compute smallest eighenvalue, corresponding eigenvector
    e_vals, e_vecs = np.linalg.eig(L)
    sorted_indices = np.argsort(e_vals)

    #Part B
    '''
    print "The smallest eigenvalue is: ", e_vals[sorted_indices[0]]
    print "The 2nd smallest eigenvalue is: ", e_vals[sorted_indices[1]]
    print "The largest eigenvalue is: ", e_vals[sorted_indices[-1]]

    print sorted_indices
    print e_vecs[:, sorted_indices[0]]
    make_plot("Smallest Eigenvector", e_vecs[:,sorted_indices[0]])
    #make_plot("Second Smallest Eigenvector", e_vecs[:, sorted_indices[1]])
    #make_plot("Largest Eigenvector", e_vecs[:, sorted_indices[-1]])
    '''

    #Part C
    e_vec_2 = e_vecs[:,sorted_indices[1]]
    e_vec_3 = e_vecs[:,sorted_indices[2]]
    '''
    plt.scatter(e_vec_2, e_vec_3)
    for i in range(100):
        for j in range(100):
            if(L[i, j] == -1):
                pass  
                
    #plt.plot([e_vec_2[0], e_vec_3[0]], [e_vec_2[1], e_vec_3[1]])


    for i in range(99):
        plt.plot([e_vec_2[i], e_vec_3[i]], [e_vec_2[i+1], e_vec_3[i+1]])
                #Draw an edge in our plot
                #plt.plot([e_vec_2[i], e_vec_3[i]], [e_vec_2[j], e_vec_3[j]])
    plt.show()
    '''


#Currently messed up
def part_1d():

    d_Laplacian = np.zeros((100, 100))
    np.random.seed(1)
    random_points = np.random.uniform(0,1,100)
    #print random_points
    for i in range(len(random_points)):
        for j in range(len(random_points)):
            if np.linalg.norm(random_points[i] - random_points[j]) <= .25 :
                if d_Laplacian[i, j] == 0:
                    d_Laplacian[i, j] = -1
                    d_Laplacian[j, i] = -1

                    d_Laplacian[i, i] += 1
                    d_Laplacian[j, j] += 1

    print d_Laplacian
    print np.sum(d_Laplacian)

    d_e_vals, d_e_vecs = np.linalg.eig(d_Laplacian)
    d_sorted_indices = np.argsort(d_e_vals)

    d_e_vec_2 = d_e_vecs[:,d_sorted_indices[1]]
    d_e_vec_3 = d_e_vecs[:,d_sorted_indices[2]]

    less_than_half_indicies = []
    for i in range(len(random_points)):
        if random_points[i] < .5:
            less_than_half_indicies.append[i]

    plt.scatter(d_e_vec_2, d_e_vec_3)
    plt.show()

def make_plot(title, vec):
    plt.ylabel('Value of Eigenvector', fontsize = 20)
    plt.xlabel('At coordinate i', fontsize = 20)  
    plt.title(title, fontsize = 24)  
    plt.plot(range(100), vec)    
    plt.show()



def build_d_laplacian():
    L = np.zeros((100,100))
    for i in range(100):
        L[i, i] = 3
        if i != 99 :
            L[i, i+1] = -1
        if i != 0 :
            L[i, i-1] = -1
        
        L[99, i] = -1
        L[i, 99] = -1

    L[99, 99] = 99
    L[0, 98] = -1
    L[98, 0] = -1
    #print L
    return L   


def build_c_laplacian():
    L = np.zeros((100,100))
    for i in range(100):
        L[i, i] = 2
        if i != 99 :
            L[i, i+1] = -1
        if i != 0 :
            L[i, i-1] = -1
        
    L[99, 0] = -1
    L[0, 99] = -1

    #print L
    return L   


def build_b_laplacian():
    L = np.zeros((100,100))
    for i in range(100):
        L[i, i] = 3
        if i != 99 :
            L[i, i+1] = -1
        if i != 0 :
            L[i, i-1] = -1
        
        L[99, i] = -1
        L[i, 99] = -1

    L[0,0] = 2
    L[98,98] = 2
    L[99, 99] = 99
    #print L
    return L   



def build_a_laplacian():
    L = np.zeros((100,100))
    for i in range(100):
        L[i, i] = 2
        if i != 99 :
            L[i, i+1] = -1
        if i != 0 :
            L[i, i-1] = -1

    L[0,0] = 1
    L[99, 99] = 1
    #print L
    return L


if __name__ == "__main__" :
    main()
