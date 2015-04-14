#p2.py
from datetime import datetime
import numpy as np
from numpy import genfromtxt
import csv
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


def main():
    #read_csvs3()
    #similarity_metrics3(cosine)
    #similarity_metrics()
    #part_1d2()

    article_matrix = read_csvs3()
    #l2_matrix = np.sqrt(np.sum(article_matrix[,:] - article_matrix[,:])**2)


    reduced_matrix = dimensionality_reduction(d=10, mat=article_matrix)
    part_1d2(article_matrix)
    #part_1d2(reduced_matrix)

    part_2b(reduced_matrix, article_matrix[2, :])



#Makes a scatter plot between article 3 and all other articles in both matricies
def part_2b(reduced_matrix, original_matrix):
    article_3 = original_matrix[2, :]
    x_coords = []
    y_coords = []
    for i in range(1000):
        #Compute distance from article_3 to each of the matricies
        x_coords.append(cosine(article_3, original_matrix[i, :]))
        y_coords.append(cosine(article_3, reduced_matrix[i, :]))

    #Plot scatter of all the 



#takes our N x K matrix, and converts it to a (N x D) matrix by using a D x K matrix of random projections
def dimensionality_reduction(d=10, mat=None):
    n, k = mat.shape
    random_matrix = np.random.normal(0, 1, (d, k))
    new_matrix = np.zeros((n, d))
    for i in range(n):
        new_matrix[i , :] = (random_matrix.dot(mat[i , :]))
    new_matrix *= 1.0/(np.sqrt(d)) #From the lecture notes, not necessarily the handout
    return new_matrix

#

#Trying again with the giant matrix
def part_1d2(article_matrix):
    article_matrix = read_csvs3()
    similarity_data = np.zeros((20,20))
    start = datetime.now()

    for i in range(20):
        for x in range(50):
            article_a = i*50 + x
            most_similar = -1
            winning_group = -1
            #print article_a
            start2 = datetime.now()

            for j in range(20):
                if(j != i):
                    for y in range(50):
                        article_b = j*50 + y
                        
                        similarity = jaccard(article_matrix[article_a, :], article_matrix[article_b, :])
                        
                        if(similarity > most_similar):
                            most_similar = similarity
                            winning_group = j
            #At the end of the article, add similarity info
            print "jaccard time: ", datetime.now() - start2
            similarity_data[i, winning_group] += 1
    print "This took : ", datetime.now()-start, " seconds"

    groups = get_group_names()
    plot_heatmat(similarity_data, groups, groups)

#Uses the giant matrix form and some hard coding, sorryyy
def similarity_metrics3(sim_metric):
    article_matrix = read_csvs3()
    similarity_data = np.zeros((20,20))
    print datetime.now().time()
    for i in range(20):
        for x in range(50):
            article_a = i*50 + x
            print article_a
            for j in range(20):
                for y in range(50):
                    article_b = j*50 + y
                    similarity_data[i, j] += sim_metric(article_matrix[article_a, :], article_matrix[article_b, :])
    groups = get_group_names()
    #print datetime.now().time()
    plot_heatmat(similarity_data, groups, groups)
    #print similarity_data
    #print similarity_data / 1000



#Plots a heatmap given a numpy array of data and labels
#http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
def plot_heatmat(data, row_labels, col_labels):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    #More stuff from the stackoverflow...
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)
    plt.show()



def get_avg_similarity(a_articles, b_articles, words_in_articles, sim_metric):
    total_similarity = 0.0
    count = 0.0
    for article_a in a_articles:
        for article_b in b_articles:
            similarity = sim_metric(words_in_articles[article_a], words_in_articles[article_b])
            count += 1.0
            total_similarity += similarity
            #print similarity
            #raw_input("")
    return total_similarity/count


#Simple function that reads CSV of group names
def get_group_names():
    groups = []
    group_file = open('p2_dataset/groups.csv', 'rb')
    for line in group_file.readlines():
        groups.append(line.strip())
    return groups


#Use the numpy stuff, vectorized code, no loops
#into each of these functions, pass 2 vectors (np arrays)
def jaccard(x,y):
    return np.sum(np.minimum(x,y))/np.sum(np.maximum(x,y))

def l2(x,y):
    return np.sqrt(np.sum((x - y)**2))

def cosine(x,y):
    return np.sum(x*y)/(np.sum(np.abs(x)) * np.sum(np.abs(y))) 




#numpy 1000 x 61100 matrix
def read_csvs3():
    data = genfromtxt('p2_dataset/data50.csv', delimiter=' ')
    matrix = np.zeros((1000,61100))
    counter = 0
    prev_article_id = 1
    for article_id, word_id, count in data:
        if(article_id != prev_article_id):
            counter += 1
        matrix[counter, word_id] = count
        prev_article_id = article_id
    return matrix



if __name__ == "__main__":
    main()




def part_1d():
    articles_in_group, word_vectors = read_csvs2()
    groups = get_group_names()

    #for each article, compute jaccard similarity between that article and all others
    similarity_data = np.zeros((20,20))
    counter = 0
    for group_num in articles_in_group:        
        for cur_article in articles_in_group[group_num]:
            counter += 1
            print "article ", counter
            most_similar = -1
            winning_group = -1
            #print article_num
            other_groups = articles_in_group.keys()[:group_num-1] + articles_in_group.keys()[group_num:]
            #print group_num
            #print other_groups
            #raw_input("")
            for other_group in other_groups:
                for other_article in articles_in_group[other_group]:
                    cur_dist = jaccard(word_vectors[cur_article], word_vectors[other_article])
                    if(cur_dist > most_similar):
                        most_similar = cur_dist
                        winning_group = other_group
            similarity_data[group_num-1, other_group-1] += 1

    plot_heatmat(similarity_data, groups, groups)
    #for article_a in all_articles



#articles in group = {1 : [1,2,3,4,5....] 2: [319, 320, 321...]}
#words in articles = {1 : [0,0,0,1,0,0,2,0....] 2 : [0,0,1,0,2,0,0,7,0,0....]}
def read_csvs2():
    label_content = open('p2_dataset/label.csv', 'rb').readlines()
    
    articles_in_group = {}
    words_in_articles = {}

    for index, line in enumerate(label_content):
        group_id = int(line.strip())
        cur_group = articles_in_group.get(group_id, [])
        if(len(cur_group) < 50):
            cur_group.append(index+1)
        articles_in_group[group_id] = cur_group


    data_matrix = genfromtxt('p2_dataset/data50.csv', delimiter=' ')
    cur_article_id = 1
    #max word id = 61100 (Technically 61067)
    cur_matrix = np.zeros((61100,1))

    for article_id, word_id, count in data_matrix:
        if(article_id != cur_article_id):
            #Save the previous matrix
            words_in_articles[cur_article_id] = cur_matrix
            cur_article_id = article_id
            cur_matrix = cur_matrix = np.zeros((61100,1))
        #add the count to the vector
        cur_matrix[word_id] = count
    #At the very end, save the last guy
    words_in_articles[cur_article_id] = cur_matrix

    return articles_in_group, words_in_articles



#Computes part 1b of the assignment. Comparing average similarity between groups
def similarity_metrics():
    articles_in_group, words_in_articles = read_csvs2()
    groups = get_group_names()
    #For each of the 20 groups, compute distance between the other 20 groups
    similarity_data = np.zeros((20,20))
    for group_a in range(1,21):
        a_articles = articles_in_group[group_a]
        for group_b in range(1,21):
            b_articles = articles_in_group[group_b]
            similarity_data[group_a-1, group_b-1] = get_avg_similarity(a_articles, b_articles, words_in_articles, cosine)
        print "iteration ", group_a, " of 20"
    plot_heatmat(similarity_data, groups, groups)




