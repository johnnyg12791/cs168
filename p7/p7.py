#p7.py
import scipy
import scipy.misc
import scipy.linalg
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import sys
import os
from cvxpy import *
from PIL import Image
from numpy import array
import copy

def main():
    if(sys.argv[1] == "1"):
        part1()
    if(sys.argv[1] == "2"):
        part2()
    if(sys.argv[1] == "3"):
        part3()

#Matrix completion
def part3():
    #np.random.seed(1) #For sanity/checking
    R = np.random.normal(0, 1, (25, 5))
    M = R.dot(R.T)

    known_entries = {}
    for row in range(M.shape[0]):
        for col in range(M.shape[1]):
            if(np.random.random() > .6):
                known_entries[(row, col)] = M[row, col]

    #Experimentation for part d
    if(sys.argv[2] == "d"):
        M_with_zeros = np.zeros_like(M)
        for key, value in known_entries.items():
            M_with_zeros[key[0], key[1]] = value        

        U, s, Vh = scipy.linalg.svd(M_with_zeros)
        k = 10
        SVD_M = U[:, :k].dot(np.diag(s[:k])).dot(Vh[:k, :])
        print np.linalg.norm(M - SVD_M)
    

    #Start the cvxpy code
    M_prime = Variable(25, 25)
    #Symmetric, positive semi-def, and matches known entries
    constraints = [M_prime == Semidef(25),
                   M_prime == M_prime.T]

    for key, value in known_entries.items():
        constraints.append(M_prime[key[0], key[1]] == value)
    # Form objective.
    obj = Minimize(trace(M_prime))
    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()

    print np.linalg.norm(M - M_prime.value)


#Recover the corrupted image!
def part2():
    img = np.array(Image.open("images/corrupted.png"), dtype=int)[:,:,0]
    Known = (img > 0).astype(int)
    print np.sum(Known)/float((Known.shape[0]*Known.shape[1]))

    print my_tv(img)
    print tv(img).value
    raw_input("")
    #From the assignment handout
    U = Variable(*img.shape)
    obj = Minimize(tv(U))
    constraints = [mul_elemwise(Known, U) == mul_elemwise(Known, img)]
    prob = Problem(obj, constraints)
    prob.solve(verbose=True, solver=SCS)
    # recovered image is now in U.value
    recovered_img = np.array(U.value)
    scipy.misc.imsave("recovered_img.png", recovered_img)

#Use info defined by websie on handout
def my_tv(U):
    total_variations = 0
    for i in range(1, U.shape[0]-1):
        for j in range(1, U.shape[1]-1):
            x = U[i+1, j] - U[i, j]
            y = U[i, j+1] - U[i, j]
            total_variations += np.sqrt(x*x + y*y)
    return total_variations
    
    


def part1():
    img = np.genfromtxt("images/wonderland-tree.txt", delimiter=1)
    n = img.shape[0] * img.shape[1]
    k = np.sum(img)
    print k/n
    np.random.seed(1)
    A = np.random.normal(0,1,(1200, 1200))
    x = np.reshape(img, (1200,1))
    #print A
    #print x
    br = A.dot(x)
    size = 600
    print "size = ", size
    x_opt = linear_program(A, br, size)
    print np.sum(np.absolute(x_opt - x))
    print np.sum(np.absolute(x_opt - x)) < .001

    #Find size
    xis = []
    yis = []
    
    for i in range(-5, 3, 1):
        print i
        r = size + i 
        x_opt = linear_program(A, br, r)
        xis.append(r)
        yis.append(np.sum(np.absolute(x_opt - x)))
    print xis
    print yis
    
    #plt.plot(xis, yis)
    #plt.ylabel('Difference')
    #plt.xlabel('R')
    #plt.title('Part 1d')
    #plt.show()

#THis is where the CXWY code goes
def linear_program(A, br, size):
    br = br[:size]
    Ar = A[:size, :]
    x = Variable(1200)
    constraints = [x >= 0,
                   Ar*x == br]
    # Form objective.
    obj = Minimize(sum_entries(x))
    # Form and solve problem.
    prob = Problem(obj, constraints)
    prob.solve()
    return x.value


if __name__ == "__main__":
    main()