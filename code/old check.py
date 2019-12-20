import math
import random
from search import *
import sys

class Simulated_Annealing():

    def __init__(self, sa_search, to_find_max = True, step = 1,temp = 10000,threshold = 1,alpha = 0.99):
        """
        :param x: starting x co-ordinate
        :param y: starting y co-ordinate
        """
        self.sa_search = sa_search
        self.point_range = point_range
        self.to_find_max = to_find_max
        self.neighbour_step = step
        self.temp = temp
        self.threshold = threshold
        self.alpha = alpha

    def finding_optimal(self):
        path = []
        # point = (search_obj.x,search_obj.y)
        path.append((search_obj.x,search_obj.y))

        # temp = 10000
        # threshold = 1
        # alpha = 0.99  # tempetaure decay value

        while (self.temp > self.threshold):
            neighbours = []
            for i in search_obj.get_neighbors(self.neighbour_step):
                neighbours.append(i)

            rand_point = neighbours[random.randint(0, len(neighbours) - 1)]
            delta_f = f(rand_point[0], rand_point[1]) - search_obj.score()
            if to_find_max:  # if we are looking for max value
                probability = math.e ** (delta_f / self.temp)
                if delta_f > 0:
                    point = rand_point
                    path.append(point)
                elif probability > (random.randint(0, 100)) / 100:
                    point = rand_point
                    path.append(point)

            if not to_find_max:  # if we are looking for min value
                probability = math.e ** (-delta_f / self.temp)
                if delta_f < 0:
                    point = rand_point
                    path.append(point)
                elif probability > (random.randint(0, 100)) / 100:
                    point = rand_point
                    path.append(point)

            temp = temp * alpha
        return path

up_to = 0
point_range = (-sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize)
point = (round((random.uniform(point_range[0], point_range[1])), up_to), round((random.uniform(point_range[2], point_range[3])), up_to))
print(point)
search_obj = Search(point[0],point[1])
sa = Simulated_Annealing(search_obj,False)
print (search_obj.x)

