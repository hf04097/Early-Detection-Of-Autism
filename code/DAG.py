import networkx as nx
from pgmpy.models import BayesianModel
import numpy as np
import random
import pandas as pd


class DAG:
    def __init__(self, ajd_matrix):
        self.matrix = np.matrix.copy(ajd_matrix)
        self.graph = nx.DiGraph(self.matrix)
        assert nx.is_directed_acyclic_graph(self.graph)

    def neighbors(self):
        rows = self.matrix.shape[0]
        cols = self.matrix.shape[1]
        neighbors = []
        dealt = set()
        total_edges = np.count_nonzero(self.matrix)
        for i in range(total_edges):
            new_matrix = np.matrix.copy(self.matrix)
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            while (row, col) in dealt:
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
            dealt.add((row, col))
            dealt.add((col, row))
            if new_matrix[row, col] == 1:
                if random.random() < 0.5:
                    new_matrix[col, row] = 1
                    new_matrix[row, col] = 0
                else:
                    new_matrix[row, col] = 0
            else:
                new_matrix[row, col] = 1
                new_matrix[col, row] = 0
            graph = nx.DiGraph(new_matrix)
            if nx.is_directed_acyclic_graph(nx.DiGraph(graph)):
                neighbors.append(new_matrix)
        return neighbors


matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
print(matrix)
gs = DAG(matrix).neighbors()
for g in gs:
    g = DAG(g)
    print(sorted(g.graph.nodes()))
    data = pd.read_csv('../dataset/test.csv', sep=',', index_col=None, header = 0)
    for col in data.columns: 
        print(col) 
    mapping = {0: 'A1', 1: 'A2', 2: 'A3'} 
    nx.relabel_nodes(g.graph, mapping, copy = False)
    print(sorted(g.graph))
    model = BayesianModel(g.graph.edges)
    print(sorted(model.nodes()))
    model.fit(data)
    for cpd in model.get_cpds():
        print(cpd)


