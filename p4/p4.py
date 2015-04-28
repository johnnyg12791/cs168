#p4.py
import random
import numpy as np
from numpy import linalg as LA
from numpy import mean, std, cov
from numpy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    if(sys.argv[1] == "1"):
        df = pd.read_csv('p4_dataset.csv')
        car_categories = list(df['Category'])
        #car_models = list(df['Model'])

        #print car_categories
        raw_data = np.genfromtxt('p4_dataset.csv', delimiter=',', skip_header=1)[:,2:]
        pca_results = PCA(raw_data)
        center_scaled_data = raw_data - mean(raw_data, 0)
        center_scaled_data /= std(center_scaled_data, 0)
        part_1d_graph(car_categories, pca_results, center_scaled_data)


    else:
        C = [.05 * i for i in range(11)]
        Y = None
        for trial in range(30):
            for c in C:
                if(sys.argv[1] == "2b"):
                    X = np.array([.001 * i for i in range(1,1001)])
                    Y = 2*X + np.random.randn(1000) * np.sqrt(c)
                    plt.title("Least Squares vs PCA on Independent data")
                else:
                    Y = [2*i/1000.0 + np.random.random() * np.sqrt(c) for i in range(1,1001)]
                    X = [i/1000.0 + np.random.random() * np.sqrt(c) for i in range(1,1001)]
                    plt.title("Least Squares vs PCA on Noisy data")

                #c on horizontal, output from recovery on vertical
                pca_recovered = plt.scatter([c], pca_recover(X,Y), c="r")
                ls_recovered = plt.scatter([c], ls_recover(X,Y), c="b")
                plt.legend([pca_recovered, ls_recovered], ("PCA", "LS"))
        plt.show()    
    #elif(sys.argv[1] == "2a"):
    #    part_2a_graph()



#Uses the first 2 principle components to look at Sports cars, suvs, minivans
def part_1d_graph(car_categories, pca_results, raw_data):
    sports_x = []
    sports_y = []
    suv_x = []
    suv_y = []
    minivan_x = []
    minivan_y = []
    v_0 = pca_results[0]
    v_1 = pca_results[1]
    for index, car_type in enumerate(car_categories):
        if(car_type == "sports"):
            sports_x.append(v_0.dot(raw_data[index,:]))
            sports_y.append(v_1.dot(raw_data[index,:]))

        if(car_type == "minivan"):
            minivan_x.append(v_0.dot(raw_data[index,:]))
            minivan_y.append(v_1.dot(raw_data[index,:]))
        if(car_type == "suv"):
            suv_x.append(v_0.dot(raw_data[index,:]))
            suv_y.append(v_1.dot(raw_data[index,:]))

    sports = plt.scatter(sports_x, sports_y,c="r")
    suv = plt.scatter(suv_x, suv_y,c="b")
    minivan = plt.scatter(minivan_x, minivan_y, c="g")
    plt.title("Car type 2D projection with PCA")
    plt.legend([sports, suv, minivan], ("Sports Car", "SUV", "Minivan"))
    plt.show()



# data[0] is the first data element, etc
def PCA(data, normalize = True):
    data -= mean(data, 0)
    if normalize:
        data /= std(data, 0)
    C = cov(data, rowvar=0) # covariance matrix
    w,V = eigh(C) # w = eigenvalues, V[:,w] = corresponding eigenvectors
    # return the eigenvectors ordered by w (in decreasing order)
    return [V[:,i] for i,e in sorted(enumerate(w), key = lambda x: x[1], reverse = True)]


# takes a vector X of xi's and a vector Y of yi's and returns the
# slope of the first component of the PCA (namely, the second coordinate divided by the first)
def pca_recover(X, Y):
    matrix = np.vstack((X,Y))
    pca = PCA(matrix.T, False)
    #hmmm...
    return pca[0][1]/pca[0][0]

#X and Y and returns the slope of the least squares fit.
#(Hint: since X is one dimensional, this takes a particularly simple form: X.dot(Y) / NormSquared(X)
def ls_recover(X, Y):
    numerator = (X - np.mean(X)).dot(Y - np.mean(Y))
    denominator = LA.norm((X - np.mean(X)))**2
    return numerator/denominator

def test_recovery():
    X = np.array([.001 * i for i in range(1,1001)])
    Y = np.array([2*xi for xi in X])

    print pca_recover(X, Y)
    print ls_recover(X, Y)

if __name__ == "__main__":
    main()
    #test_recovery()

