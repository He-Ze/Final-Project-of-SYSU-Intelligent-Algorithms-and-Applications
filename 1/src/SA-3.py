import matplotlib.pyplot as plt
import random
import math
import re

class city:
    def __init__(self, index, x, y):
        self.cityIndex = index
        self.x = x
        self.y = y

def initCities(filePath):
    with open(filePath) as file:
        cities = []
        begin = False
        for line in file.readlines()[0:-1]:
            if line.startswith('EOF'):
                break
            if line.startswith('NODE_COORD_SECTION'):
                begin = True
            elif begin == True:
                info = re.split('[ ]+', line.strip())
                cities.append(city(info[0], float(info[1]), float(info[2])))
    return cities

def calDistance(path):
    dis = 0.0
    for i in range(len(path) - 1):
        dis += distance[path[i]][path[i+1]]
    return dis


T = 4000
alpha = 0.99
distance = []
cities = initCities("../kroA150.tsp")
numOfCity = len(cities)

for i in range(numOfCity):
    dis = []
    for j in range(len(cities)):
        dis.append(int(((cities[i].x - cities[j].x)**2 + (cities[i].y - cities[j].y)**2)**0.5))
    distance.append(dis)

path = [i for i in range(numOfCity)]
random.shuffle(path)path.append(path[0])
dis = calDistance(path)

while(T > 0.001):
    for i in range(1000):
        first = random.randint(1, len(path)-4)
        second = random.randint(first+1, len(path)-3)
        third = random.randint(second+1, len(path)-2)
        dE = distance[path[first-1]][path[third]] + distance[path[second+1]][path[first]] + distance[path[second]][path[third+1]]- distance[path[first-1]][path[first]] - distance[path[second]][path[second+1]] - distance[path[third]][path[third+1]]
        if dE < 0 or random.random() < math.exp(-dE / T):
            path[first:third+1]= path[third:second:-1]+path[first:second+1]
            dis = dis + dE
    T *= alpha

    plt.clf()
    ax = plt.axes()
    ax.set_title('Distance: ' + str(dis))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for n in range(len(path)-1):
        plt.plot([cities[path[n]].x, cities[path[n + 1]].x], [cities[path[n]].y, cities[path[n+1]].y], '-ro')
    plt.pause(0.001)

# output
print(path)
print(dis)
plt.clf()
ax = plt.axes()
ax.set_title('Distance: ' + str(dis))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
for n in range(len(path)-1):
    plt.plot([cities[path[n]].x, cities[path[n + 1]].x], [cities[path[n]].y, cities[path[n+1]].y], '-ro')
plt.show() 