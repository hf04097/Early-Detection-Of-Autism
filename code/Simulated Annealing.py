import math
import random
from search import *
import sys
import matplotlib.pyplot as plt


def decay_func(x, factor = 50, decay_factor = 50):
    return factor * (math.e ** -( (x) / decay_factor))

def probability_function(change, temperature):
    return math.e ** (change/temperature)

class Simulated_Annealing():

    def __init__(self, sa_search, find_max = True, step = 0.1 ,temp=1000,alpha=0.99,threshold = 1):
        """
        :param x: starting x co-ordinate
        :param y: starting y co-ordinate
        """
        self.sa_search = sa_search
        self.to_find_max = find_max
        self.step = step
        # self.decay_factor = df
        self.temperature = temp
        self.threshold = threshold
        self.alpha = alpha

    def finding_optimal(self):
        iteration = 0
        iter_list = [iteration]
        func_output = []
        search_end = False
        visited = set()
        path = []
        # temperature = decay_func(iteration, decay_factor=self.decay_factor)

        while not search_end:
            neighbors = []
            check_neighbours = self.sa_search.get_neighbors(self.step)
            for i in check_neighbours:
                if i not in visited:
                    neighbors.append(i)
            # for i in neighbors:
            #     print (self.sa_search.x,self.sa_search.y,' : ',i.x,i.y)
            #     # print (i.x,i.y)
            # print ('\n')

            if not self.to_find_max:
                change = -1
            else:
                change = 1
            max_dir = -1

            while max_dir == -1:
                index = random.randint(0, len(neighbors) - 1)
                output = change * (self.sa_search.score() - neighbors[index].score())
                if output > 0:
                    # print ('yes')
                    max_dir = index
                else:
                    # print ('else')
                    prob = 100 * probability_function(output, self.temperature)
                    random_num = random.uniform(0, 100)
                    if random_num < prob:
                        # print ('gogo')
                        max_dir = index


            self.temperature = self.alpha*self.temperature
            func_output.append(self.sa_search.score())
            next_point = neighbors[max_dir]
            visited.add(next_point)
            path.append(next_point)
            iteration += 1
            iter_list.append(iteration)
            # temperature = decay_func(iteration, decay_factor=self.decay_factor)
            self.sa_search = next_point
            # print (self.sa_search.x,self.sa_search.y)
            if self.temperature < self.threshold:
                func_output.append(self.sa_search.score())
                search_end = True

        plt.plot(iter_list, func_output)
        plt.xlabel('iterations')
        plt.ylabel('func value')
        plt.show()
        return path

up_to = 0
# point_range = (-sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize)
point_range = (-5,5,-5,5)
point = (round((random.uniform(point_range[0], point_range[1])), up_to), round((random.uniform(point_range[2], point_range[3])), up_to))
# point = (3.0,1.0)
# print(point)
search_obj = Search(point[0],point[1])
sa = Simulated_Annealing(search_obj,False,1)
path = sa.finding_optimal()
lst = path[-1]
print (lst.x,lst.y)
# for i in lst:
#     print (i.x,i.y)