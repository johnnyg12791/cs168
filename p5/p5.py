#p5.py
import scipy.misc
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os


def main():
    part = sys.argv[1]
    if(part == "1"):
        part1()
    if(part == "2"):
        part2()
    if(part == "3"):
        part3()



def part3():
    T = np.random.uniform(-1,1,(10,100))
    R = np.random.normal(0,1,(50, 10))
    M = R.dot(T)
    U, s, Vh = scipy.linalg.svd(M)
    #print s #(or plot S)
    np.set_printoptions(threshold='nan')
    #part B
    M_hat = M + np.random.normal(0, .1, (50, 100))

    #randomly set 500 of these to 0
    num_removed = 0
    while(num_removed < 3000):
        row = np.random.uniform(0, 50)
        col = np.random.uniform(0, 100)
        if(M_hat[row, col] != 0):
            M_hat[row, col] = np.average(M)
            num_removed += 1

    #print M_hat

    M_prime = get_SVD_reduced(M_hat, 10)

    print np.linalg.norm(M_prime - M)
    print np.linalg.norm(M_hat - M)


#for each image in the folder
    #read in image
def part2():

    img_data = np.zeros((10,22500))
    for row, filename in enumerate(os.listdir('./p5_dataset/')):
        img = scipy.misc.imread('./p5_dataset/' + filename)[:, :, 0].reshape(1, 22500)
        img_data[row, :] = img

    #center data
    centered = img_data/img_data.sum(axis=0, keepdims=True)
    #run SVD
    #get first component
    U, s, Vh = scipy.linalg.svd(centered)

    if(sys.argv[2] == "a"):
        plt.imshow(Vh[0].reshape(180, 125), cmap = cm.Greys_r)
        plt.show()   
    if(sys.argv[2] == "b"):
        #Do projections
        barack = []
        michelle = []
        for row in range(centered.shape[0]):
            if(row < 5):
                barack.append(Vh[0].dot(centered[row]))
            else:
                michelle.append(Vh[0].dot(centered[row]))

        #Now plot the points
        b = plt.scatter(range(1,6),barack,c="r")
        r = plt.scatter(range(6,11),michelle,c="b")
        plt.legend([b, r], ["Barack", "Michelle"])
        plt.axis((0, 12, 0, -25))
        plt.show()


def get_SVD_reduced(matrix, k):
    U, s, Vh = scipy.linalg.svd(matrix)

    return U[:, :k].dot(np.diag(s[:k])).dot(Vh[:k, :])


def part1():
    k = int(sys.argv[2])
    img = scipy.misc.imread("p5_image.gif")

    inverted = True
    if(inverted):
        img = (img * -1) + 1

    reduced_matrix = get_SVD_reduced(img, k)

    if(inverted):
        reduced_matrix = (reduced_matrix * -1) + 1

    plt.imshow(reduced_matrix, cmap = cm.Greys_r)
    plt.show()   



if __name__ == "__main__":
    main()
