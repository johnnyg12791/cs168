#p3.py
import math
import random
import copy
import numpy as np
import sys
import matplotlib.pyplot as plt


def main():
    t = int(sys.argv[1])
    #Contest("data/parksContest.csv", t)
    Assignment("data/parks.csv", t, "D", 70) #Set to "D" for that 1D, also set 0 to 70 for part 2A


#This runs the code for the Assignment, can be modified for parts B, D or 2A
def Assignment(filename, t, part, c):
    parks_data = read_parks_csv(filename)
    num_trials = 10    

    best_route_tracker_dict = {}
    for i in range(num_trials):    
        best_route_tracker, best_route = MCMC(maxiter=10000, data=parks_data, T=t, c=c, part=part)
        best_route_tracker_dict[i] = best_route_tracker
        best_dist = best_route_tracker[len(best_route_tracker) -1]
        print "best dist for trial ", i, " is : ", best_dist
    plot_figure(best_route_tracker_dict, t)
    #plot_best_route


'''
Runs our MonteCarlo MarkovChain Model
Input: number of max iterations
Input: dictionary - data {"A" : (1,3), "B" : (4, 8)...}
Input: number T temperature
Input: number c annealing factor (if 0, no annealing)
Input: string "B" to take 2 parks next to eachother in route
'''
def MCMC(maxiter=10000, data=None, T=10, c=0, part="B"):
    route_tracker = []
    route = list(data.keys())
    random.shuffle(route)
    best_route = route
    best_dist = route_dist(route, data)
    for i in xrange(maxiter):
        if(c > 0):
            T = c / math.sqrt(i+1)

        first, second = 0, 0
        if(part == "B"):
            first = random.randint(0, len(route)-1)
            second = (first + 1) % len(route)
        else: #Assumed to be D if not B
            first, second = random.sample(range(len(route)-1), 2)

        new_route = copy.deepcopy(route)

        #Switch the indicies, then compute new distance
        new_route[first], new_route[second] = new_route[second], new_route[first]
        new_dist = route_dist(new_route, data)
        delta_dist = new_dist - route_dist(route, data)
        if(delta_dist < 0 or (T > 0 and random.random() < np.exp(-delta_dist/T))):
            route = new_route
        if(route_dist(route, data) < best_dist):
            best_route = route
            best_dist = route_dist(best_route, data)
        #print "Best distance is: ", best_dist
        route_tracker.append(route_dist(route, data))
    return route_tracker, best_route


def plot_best_route(route, parks_data):
    x_coords = []
    y_coords = []
    #use the x coordinate for longitude and the y coordinate for latitude
    for park in route:
        x_coords.append(parks_data[park][0])
        y_coords.append(parks_data[park][1])
    x_coords.append(parks_data[route[0]][0])
    y_coords.append(parks_data[route[0]][1])

    plt.ylabel('Longitude')
    plt.xlabel('Latitude')
    plt.title('Best Route: ' + str(round(route_dist(route, parks_data), 2))) 
    plt.plot(x_coords, y_coords, marker='o')#, linestyle='--', color='r')
    plt.show()


#Takes in a dict of y_coords
#Plot each of those as a line, with range(len(dict[0])) as x_coords
def plot_figure(best_route_tracker_dict, t):
    plt.ylabel('distance')
    plt.xlabel('iteration')
    #title = "Temperature = " + str(t)
    title = "Annealing, c=70"
    plt.title(title)
    for key, value in best_route_tracker_dict.items():
        plt.plot(value)
    
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,100,600))
    
    plt.show()


'''
Reads in CSV of:
HEADER
park1, lat, long
park2, lat, long

into a dict of {park1 : (lat, long), park2: (lat, long)}
'''
def read_parks_csv(filename):
    park_data = {}
    f = open(filename)
    content = f.readlines()[1:] #Skip header
    for line in content:
        split_line = line.strip().split(",")
        park_data[split_line[0]] = (float(split_line[1]), float(split_line[2]))
    return park_data

#Calculates the distance between a route (list of parks)
def route_dist(route, data):
    total_dist = 0
    cur_park = route[0]
    for i in range(1, len(route)):
        next_park = route[i]
        total_dist += park_dist(data, cur_park, next_park)
        cur_park = next_park
    #Route has to return to the start
    return (total_dist + park_dist(data, cur_park, route[0]))

#Calculates the distance between two parks
def park_dist(data, park1, park2):
    return dist(data[park1][0], data[park1][1], data[park2][0], data[park2][1])

#Calculates distance given lats and longs
def dist(lat1, long1, lat2, long2):
    return math.sqrt((lat1-lat2)**2 + (long1-long2)**2)

if __name__ == "__main__":
    main()


#This is the contest code, works with a slightly different MCMC function, 
#keeps track of more interesting stats
def Contest(filename, t):
    parks_data = read_parks_csv(filename)
    #This is our best distance so far, which is = 416.8
    print route_dist(['Rocky Mountain', 'Great Sand Dunes', 'Black Canyon of the Gunnison', 'Mesa Verde', 'Arches', 'Canyonlands', 'Capitol Reef', 'Bryce Canyon', 'Grand Canyon', 'Zion', 'Great Basin', 'Grand Teton', 'Yellowstone', 'Glacier', 'North Cascades', 'Mount Rainier', 'Crater Lake', 'Lassen Volcanic', 'Redwood', 'Olympic', 'Glacier Bay', 'Wrangell \xe2\x80\x93St. Elias', 'Kenai Fjords', 'Denali', 'Gates of the Arctic', 'Kobuk Valley', 'Lake Clark', 'Katmai', 'Haleakal\xc4\x81', 'Hawaii Volcanoes', 'American Samoa', 'Channel Islands', 'Pinnacles', 'Yosemite', 'Kings Canyon', 'Sequoia', 'Death Valley', 'Joshua Tree', 'Saguaro', 'Petrified Forest', 'Guadalupe Mountains', 'Carlsbad Caverns', 'Big Bend', 'Hot Springs', 'Mammoth Cave', 'Great Smoky Mountains', 'Congaree', 'Dry Tortugas', 'Everglades', 'Biscayne', 'Virgin Islands', 'Acadia', 'Shenandoah', 'Cuyahoga Valley', 'Isle Royale', 'Voyageurs', 'Theodore Roosevelt', 'Badlands', 'Wind Cave'], parks_data)
    #Statistics
    absolute_best_dist = 1000
    absolute_best_route = []
    total_distances = 0
    num_trials = 10

    for i in range(num_trials):
        best_route = MCMC_Contest(maxiter=2000000, data=parks_data, T=t, c=200)
        best_dist = route_dist(best_route, parks_data)

        print "best dist for trial ", i, " is : ", best_dist
        total_distances += best_dist
        if(best_dist < absolute_best_dist):
            absolute_best_dist = best_dist
            absolute_best_route = best_route

    plot_best_route(absolute_best_route, parks_data)
    print absolute_best_route
    print "avg best distance: ", total_distances/num_trials
    #plot_figure(best_route_tracker_dict, t)


#Trying some strategies for the contest
def MCMC_Contest(maxiter=1000, data=None, T=10, c=0):
    #route = list(data.keys())
    #random.shuffle(route)
    route = ['Rocky Mountain', 'Great Sand Dunes', 'Black Canyon of the Gunnison', 'Mesa Verde', 'Arches', 'Canyonlands', 'Capitol Reef', 'Bryce Canyon', 'Grand Canyon', 'Zion', 'Great Basin', 'Grand Teton', 'Yellowstone', 'Glacier', 'North Cascades', 'Mount Rainier', 'Crater Lake', 'Lassen Volcanic', 'Redwood', 'Olympic', 'Glacier Bay', 'Wrangell \xe2\x80\x93St. Elias', 'Kenai Fjords', 'Denali', 'Gates of the Arctic', 'Kobuk Valley', 'Lake Clark', 'Katmai', 'Haleakal\xc4\x81', 'Hawaii Volcanoes', 'American Samoa', 'Channel Islands', 'Pinnacles', 'Yosemite', 'Kings Canyon', 'Sequoia', 'Death Valley', 'Joshua Tree', 'Saguaro', 'Petrified Forest', 'Guadalupe Mountains', 'Carlsbad Caverns', 'Big Bend', 'Hot Springs', 'Mammoth Cave', 'Great Smoky Mountains', 'Congaree', 'Dry Tortugas', 'Everglades', 'Biscayne', 'Virgin Islands', 'Acadia', 'Shenandoah', 'Cuyahoga Valley', 'Isle Royale', 'Voyageurs', 'Theodore Roosevelt', 'Badlands', 'Wind Cave']
    best_route = route
    best_dist = route_dist(route, data)
    for i in xrange(maxiter):
        if(c > 0):
            T = c / math.sqrt(i+1)
        #pick a random int between 0 and len(route)
        first, second, third, fourth = random.sample(range(len(route)-1), 4)

        #Randomly choose 4 of the indicies
        rand_list = random.sample(range(len(route)-1), 4)
        orig_list = copy.deepcopy(rand_list)

        #Try to shuffle those 4 indicies 10 times. Sometimes this has the effect of just swapping 2 parks, or 3. Occasionally all 4
        for j in range(10):
            rt = copy.deepcopy(route)
            random.shuffle(orig_list)
            #Randomly permute the 4 indicies originally choosen
            rt[rand_list[0]], rt[rand_list[1]], rt[rand_list[2]], rt[rand_list[3]] = rt[orig_list[0]], rt[orig_list[1]], rt[orig_list[2]], rt[orig_list[3]]
            #Check temperature/if we get a lower distance
            new_dist = route_dist(rt, data)
            delta_dist = new_dist - route_dist(route, data)
            if(delta_dist < 0 or (T > 0 and random.random() < np.exp(-delta_dist/T))):
                route = rt
            #set new best route if found
            if(route_dist(route, data) < best_dist):
                best_route = route
                best_dist = route_dist(best_route, data)
    return best_route
