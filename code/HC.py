import sys
import matplotlib.pyplot as plt
from DAG import *
import math
from search import *

class HillClimbing():

    def __init__(self, sa_search, find_max = True, step = 0.1,max_iter = 200):
        """
        :param x: starting x co-ordinate
        :param y: starting y co-ordinate
        """
        self.sa_search = sa_search
        self.to_find_max = find_max
        self.step = step
        self.max_iter = max_iter

    def finding_optimal(self, debug = False):
        iteration = 0
        iter_list = []
        func_output = []
        search_end = False
        visited = set()
        path = []
        # temperature = decay_func(iteration, decay_factor=self.decay_factor)
        if self.to_find_max:
            best_score = -math.inf
        else:
            best_score = math.inf
        best_graph = None

        while not search_end:
            if debug and iteration % 10 == 0:
                print(iteration, end=' ')

            neighbors = []
            check_neighbours = self.sa_search.get_neighbors()
            for i in check_neighbours:
                if i not in visited:
                    neighbors.append(i)
            if not self.to_find_max:
                change = -1
            else:
                change = 1
            max_dir = -1

            current = self.sa_search.score()
            scores = [round(change * (- self.sa_search.score() + neighbor.score()),3) for neighbor in neighbors]
            max_dir = scores.index(max(scores))
            if scores[max_dir] < 0 or iteration > self.max_iter:
                search_end = True

            func_output.append(current)
            next_point = neighbors[max_dir]
            visited.add(next_point)
            path.append(next_point)
            iteration += 1
            iter_list.append(iteration)
            # temperature = decay_func(iteration, decay_factor=self.decay_factor)
            self.sa_search = next_point
            # print (self.sa_search.x,self.sa_search.y)

        print(len(iter_list))
        print(len(func_output))
        plt.plot(iter_list, func_output)
        plt.xlabel('iterations')
        plt.ylabel('func value')
        plt.show()
        return path[-1], scores[max_dir], path

if __name__ == '__main__':
    #freeze_support()
    up_to = 0
    # point_range = (-sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize)
    point_range = (-5,5,-5,5)
    point = (round((random.uniform(point_range[0], point_range[1])), up_to), round((random.uniform(point_range[2], point_range[3])), up_to))
    # point = (3.0,1.0)
    # print(point)
    search_obj = Search(point[0],point[1])
    sa = HillClimbing(search_obj,True,1)
    obj, score, path = sa.finding_optimal(debug = True)