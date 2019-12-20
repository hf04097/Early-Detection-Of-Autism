import math
import random
from search import *
import sys
import matplotlib.pyplot as plt
from DAG import *


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

    def finding_optimal(self, debug = False):
        iteration = 0
        iter_list = [iteration]
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
            # for i in neighbors:
            #     print (self.sa_search.x,self.sa_search.y,' : ',i.x,i.y)
            #     # print (i.x,i.y)
            # print ('\n')

            if not self.to_find_max:
                change = -1
            else:
                change = 1
            max_dir = -1
            current = self.sa_search.score()
            while max_dir == -1:
                index = random.randint(0, len(neighbors) - 1)
                neighbor_score = neighbors[index].score()
                output = change * (-current  + neighbor_score)
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

            if self.to_find_max:
                if neighbor_score > best_score:
                    best_graph = neighbors[max_dir]
                    best_score = neighbor_score
            else:
                if neighbor_score < best_score:
                    best_graph = neighbors[max_dir]
                    best_score = neighbor_score


            self.temperature = self.alpha*self.temperature
            func_output.append(current)
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
        return best_graph, best_score, path

if __name__ == '__main__':
    #freeze_support()
    up_to = 0
    # point_range = (-sys.maxsize, sys.maxsize, -sys.maxsize, sys.maxsize)
    point_range = (-5,5,-5,5)
    point = (round((random.uniform(point_range[0], point_range[1])), up_to), round((random.uniform(point_range[2], point_range[3])), up_to))
    # point = (3.0,1.0)
    # print(point)
    search_obj = Search(point[0],point[1])
    sa = Simulated_Annealing(search_obj,False,1)
    obj, score = sa.finding_optimal()
    #lst = path[-1]
    print (obj)
    # for i in lst:
    #     print (i.x,i.y)

    _, cols = read_data('../dataset/Toddler Autism dataset July 2018.csv', train_flag= True)
    cols = len(cols)
    matrix = np.zeros((cols, cols))
    for i in range(cols - 1):
        matrix[i, cols - 1] = 1
    test = DAG(matrix, name = '../dataset/Toddler Autism dataset July 2018.csv')
    sa = Simulated_Annealing(test, True, 1, alpha = 0.95)
    best_sol, best_score = sa.finding_optimal()

    #print(best_sol.test_MSE())