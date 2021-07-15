from sys import argv, float_info
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


def initPopulation(cities, numOfCity):
    population = []
    individual = [i for i in range(numOfCity)]
    for r in range(int(POPSIZE * 2 / 10)):
        random.shuffle(individual)
        population.append(individual[:])
    for r in range(int(POPSIZE - len(population))):
        start = random.randint(0, numOfCity-1)
        gIndividual = []
        gIndividual.append(start)
        j = 1
        while j < numOfCity:
            mixDis = float_info.max 
            i, bestId = 0, 0
            while i < numOfCity:
                if (i not in gIndividual) and i != gIndividual[-1] and distance[gIndividual[-1]][i] < mixDis:
                    bestId = i
                    mixDis = distance[gIndividual[-1]][i]
                i += 1
            j = j + 1
            gIndividual.append(bestId)
        population.append(gIndividual[:]) 
    random.shuffle(population)
    return population

def calFitness(individual):
    fitness = 0.0
    for i in range(len(individual) - 1):
        fitness += distance[individual[i]][individual[i+1]]
    fitness += distance[individual[len(individual)-1]][individual[0]]
    return fitness

def select(population, numOfCity):
    newPopulation = []
    best = float_info.max
    bestId = 0
    fitness = []
    sumOfFitness = 0.0
    for i in range(POPSIZE):
        fit = calFitness(population[i])
        fitness.append(1 / fit)
        sumOfFitness += 1 / fit
        if (best > fit) :
            best = fit
            bestId = i
    newPopulation.append(population[bestId])
    cumPro = []
    for i in range(POPSIZE):
        if i == 0:
            cumPro.append(fitness[i] / sumOfFitness)
        else:
            cumPro.append(fitness[i] / sumOfFitness + cumPro[i-1])       
    for i in range(POPSIZE-1):
        pro = random.random()
        for j in range(POPSIZE):
            if cumPro[j] >= pro:
                newPopulation.append(population[j])
                break
    return newPopulation

def crosscover(population, numOfCity):    subPopulation = []
    for i in range(POPSIZE):
        if random.random() <= PXOVER:
            chromosomeFir = random.randint(0, POPSIZE - 1)
            chromosomeSec = random.randint(0, POPSIZE - 1)
            while chromosomeFir == chromosomeSec:
                chromosomeSec = random.randint(0, POPSIZE - 1)
            start = random.randint(0, numOfCity - 2)
            end = random.randint(start + 1, numOfCity - 1)
            newIndividual_i = []
            newIndividual_j = []
            k = 0
            for j in range(numOfCity):
                if j >= start and j < end:
                    newIndividual_i.append(population[chromosomeFir][j])
                    j = end
                else:
                    while k < numOfCity:
                        if population[chromosomeSec][k] not in population[chromosomeFir][start:end]:
                            newIndividual_i.append(population[chromosomeSec][k])
                            k += 1
                            break
                        k += 1
            k = 0
            for j in range(numOfCity):
                if population[chromosomeSec][j] in population[chromosomeFir][start:end]:
                    newIndividual_j.append(population[chromosomeSec][j])
                else:
                    if k == start:
                        k = end
                    newIndividual_j.append(population[chromosomeFir][k])
                    k += 1
            subPopulation.append(newIndividual_i[:])
            subPopulation.append(newIndividual_j[:])
    subPopulation.sort(key = lambda x: calFitness(x))
    for i in range(len(subPopulation)):
        for j in range(POPSIZE):
            if calFitness(subPopulation[i]) < calFitness(population[j]):
                population[j] = subPopulation[i]
                break
    return population

def mutate(population, numOfCity):
    for i in range(len(population)):
        if random.random() <= PMUTATION:
            geneFir = random.randint(1,numOfCity-2)
            geneSec = random.randint(geneFir+1, numOfCity-1)
            population[i][geneFir:geneSec] = population[i][geneSec-1:geneFir-1:-1]
        if random.random() <= PMUTATION:
            geneFir = random.randint(0,numOfCity-1)
            geneSec = random.randint(0, numOfCity-1)
            while geneFir == geneSec:
                geneSec = random.randint(0, numOfCity-1)
            population[i][geneFir], population[i][geneSec] = population[i][geneSec], population[i][geneFir]
    return population

def localSearch(population, numOfCity):
    for i in range(len(population)):
        best = population[i][:]
        for _ in range(100):
            first = random.randint(1, numOfCity - 2)
            second = random.randint(first + 1, numOfCity - 1)
            population[i][first:second] = population[i][second-1:first-1:-1]
            if calFitness(best) > calFitness(population[i]):
                best = population[i][:]
        population[i] = best
    return population


POPSIZE = 50
MAXGENS = 1000
PXOVER = 0.9
PMUTATION = 0.2
distance = []
cities = initCities("../kroA150.tsp")
numOfCity = len(cities)

for i in range(len(cities)):
    dis = []
    for j in range(len(cities)):
        dis.append(int(((cities[i].x - cities[j].x)**2 + (cities[i].y - cities[j].y)**2)**0.5))
    distance.append(dis)

population = initPopulation(cities, numOfCity)
curGen = 0 
while curGen < MAXGENS:
    random.shuffle(population)
    population = select(population, numOfCity)
    population = crosscover(population, numOfCity)
    population = mutate(population, numOfCity)
    population = localSearch(population, numOfCity)
    population.sort(key = lambda x: calFitness(x))
    print("Current: ", curGen)
    plt.clf()
    ax = plt.axes()
    ax.set_title('Distance: ' + str(calFitness(population[0])))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for n in range(numOfCity - 1):
        plt.plot([cities[population[0][n]].x, cities[population[0][n + 1]].x], [cities[population[0][n]].y, cities[population[0][n+1]].y], '-ro')
    plt.plot([cities[population[0][-1]].x, cities[population[0][0]].x], [cities[population[0][-1]].y, cities[population[0][0]].y], '-ro')
    plt.pause(0.001)
    curGen += 1
population[0].append(population[0][0])
print(population[0])
print(calFitness(population[0]))
plt.clf()
ax = plt.axes()
ax.set_title('Distance: ' + str(calFitness(population[0])))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
for n in range(numOfCity - 1):
    plt.plot([cities[population[0][n]].x, cities[population[0][n + 1]].x], [cities[population[0][n]].y, cities[population[0][n+1]].y], '-ro')
plt.plot([cities[population[0][-1]].x, cities[population[0][0]].x], [cities[population[0][-1]].y, cities[population[0][0]].y], '-ro')
plt.show()
