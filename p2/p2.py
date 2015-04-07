#p2.py

import numpy as np


def main():
    similarity_metrics()


def similarity_metrics():
    bag_of_words = read_csv('p2_dataset/data50.csv')
    groups = read_groups('p2_dataset/group.csv')
    labels = read_labels('p2_dataset/label.csv')

    print "Jaccard similarity: ", jaccard(x,y)
    print "L2 similarity     : ", l2(x,y)
    print "Cosine similarity : ", cosine(x,y)



#Use the numpy stuff, vectorized code, no loops
def jaccard(x,y):
    return np.min(x,y)/np.max(x,y)



def l2(x,y):
    return np.sqrt(np.sum((x - y)**2))


def cosine(x,y):
    return x*y/(np.sum(np.abs(x)) * np.sum(np.abs(y))) 


def dimension_reduction():
    pass

if __name__ == "__main__":
    main()