#p1.py
#N bins and N balls, throw each ball into a bin 1 by 1.


import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def main():
    N = 100000 
    num_sims = 30 
    max_vals = []
    for i in xrange(num_sims):
        #max_vals.append(strategy_1(N))
        #max_vals.append(strategy_2(N))
        #max_vals.append(strategy_3(N))
        max_vals.append(strategy_4(N))
    print max_vals
    plt.hist(max_vals, bins=5, range=(1.5, 6.5))
    plt.show()

# Select one of the N bins uniformly at random, and place the current ball in it.
def strategy_1(N):
    buckets = [0 for x in range(0,N)]
    for i in xrange(N+1):
        bucket = random.randint(0, N-1)
        buckets[bucket] += 1

    return max(buckets)

'''
Select two of the N bins uniformly at random (either with or without replacement), and look at how
many balls are already in each. If one bin has strictly fewer balls than the other, place the current
ball in that bin. If both bins have the same number of balls, pick of the two at random and place the
current ball in it.
'''
def strategy_2(N):
    buckets = [0 for x in range(0,N)]
    for i in xrange(N+1):
        bucket1 = random.randint(0, N-1)
        bucket2 = random.randint(0, N-1)
        #add it to bucket with less

        if(buckets[bucket1] > buckets[bucket2]):
            buckets[bucket2] += 1
        else:
            buckets[bucket1] += 1
    return max(buckets)

#Same as the previous strategy, except choosing three bins at random rather than two
def strategy_3(N):
    buckets = [0 for x in range(0,N)]
    for i in xrange(N+1):
        bucket1 = random.randint(0, N-1)
        bucket2 = random.randint(0, N-1)
        bucket3 = random.randint(0, N-1)
        #add it to bucket with less

        if(buckets[bucket1] < buckets[bucket2] and buckets[bucket1] < buckets[bucket3]):
            buckets[bucket1] += 1
        elif(buckets[bucket2] < buckets[bucket1] and buckets[bucket2] < buckets[bucket3]):
            buckets[bucket2] += 1
        else:
            buckets[bucket3] += 1
    return max(buckets)

'''
Select two bins as follows: the first bin is selected uniformly from the first N/2 bins, and the second
uniformly from the last N/2 bins. (You can assume that N is even.) If one bin has strictly fewer balls
than the other, place the current ball in that bin. If both bins have the same number of balls, place
the current ball (deterministically) in the first of the two bins
'''
def strategy_4(N):
    buckets = [0 for x in range(0,N)]
    for i in xrange(N+1):
        bucket1 = random.randint(0, N/2)
        bucket2 = random.randint(N/2, N-1)
        #add it to bucket with less

        if(buckets[bucket1] > buckets[bucket2]):
            buckets[bucket2] += 1
        else:
            buckets[bucket1] += 1
    return max(buckets)



if __name__ == "__main__":
    main()