import networkx as nx
from pgmpy.models import BayesianModel
import numpy as np
import random
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import K2Score
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from pgmpy.inference import VariableElimination
from multiprocessing import Pool, freeze_support
from functools import partial

class DAG:
    def __init__(self, ajd_matrix, name = '../dataset/test.csv'):
        self.matrix = np.matrix.copy(ajd_matrix)
        self.graph = nx.DiGraph(self.matrix)
        self.trained = False
        self.model = None
        self.file_name = name
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
            neighbor = DAG(new_matrix)
            if nx.is_directed_acyclic_graph(neighbor.graph) and nx.is_weakly_connected(neighbor.graph):
                neighbors.append(neighbor)
        return neighbors

    def fit_bayesian(self):
        assert not self.trained
        print("reading data")
        data, names = read_data(self.file_name, train_flag = True)
        mapping = {i:names[i] for i in range(len(names))}
        nx.relabel_nodes(self.graph, mapping, copy=False)
        print(self.graph.edges)
        print("fitting the model")
        self.model = BayesianModel(self.graph.edges)
        print("fitting model on data")
        self.model.fit(data, estimator= BayesianEstimator, prior_type = "K2")
        self.trained = True


    def get_row_prediction(self, row, model, col_names):
        col = col_names
        final_predictions_actual = []
        final_predications_model = []
        to_pred = []
        label = random.random() < 0.5  # including the label with RHS
        if label:
            to_pred.append('Class/ASD Traits ')
            final_predictions_actual.append(row[1].values[-1])
        variables_to_hide = int(round(np.random.normal(2), 0))
        for i in range(variables_to_hide):
            column = random.randint(0, len(col) - 2)
            to_pred.append(col[column])
            final_predictions_actual.append(row[1].values[column])
        if to_pred:
            evidence = {row[1].index[i]: row[1].values[i] for i in range(len(row[1].index)) if
                        row[1].index[i] not in to_pred}
            prediction = (model.query(variables=to_pred, evidence=evidence, joint=False, show_progress=False))
            states = prediction[to_pred[0]].state_names
            for i in to_pred:
                final_predications_model.append(
                    (states[i][np.where(prediction[i].values == max(prediction[i].values))[0][0]]))
            return (accuracy_score(final_predictions_actual, final_predications_model))
        return 1

    def get_error(self, data):
        model_inference = VariableElimination(self.model)
        cols = list(data.columns)
        print("covering rows to list")
        rows = list(data.iterrows())
        print("done")
        with Pool() as pool:
            results = pool.map(partial(self.get_row_prediction, model = model_inference, col_names = cols), rows, chunksize= len(rows))
        print("got results", results)
        print(np.nanmean(results))

        """
        col = list(data.columns)
        final_predictions_actual = []
        final_predications_model = []
        for row in data.iterrows():
            to_pred = []
            label = random.random() < 0.5 #including the label with RHS
            if label:
                to_pred.append('Class/ASD Traits ')
                final_predictions_actual.append(row[1].values[-1])
            variables_to_hide = int(round(np.random.normal(2),0))
            for i in range(variables_to_hide):
                column = random.randint(0, len(col) - 2)
                to_pred.append(col[column])
                final_predictions_actual.append(row[1].values[column])
            if to_pred:
                evidence = {row[1].index[i]: row[1].values[i] for i in range(len(row[1].index)) if row[1].index[i] not in to_pred}
                prediction = (model_inference.query(variables = to_pred, evidence = evidence, joint = False, show_progress= False))
                states = prediction[to_pred[0]].state_names
                for i in to_pred:
                    final_predications_model.append((states[i][ np.where(prediction[i].values == max(prediction[i].values))[0][0]]))

        print(f1_score(final_predictions_actual, final_predications_model))
        """

    def score(self):
        if not self.trained:
            print("model not training, going to training")
            self.fit_bayesian()
            print("training the model")
        data, _ = read_data(self.file_name, verification_flag = True)
        print(data.shape)
        print("now computing score")
        self.get_error(data)
        #return K2Score(data).score(self.model)

    def __hash__(self):
        return hash(self.graph)

DATA_READ = False
TEST = None
TRAIN = None
VALID = None
COL_NAMES = None

def read_data(file_name, train_flag = False, test_flag = False, verification_flag = False):
    global DATA_READ, TEST, TRAIN, VALID, COL_NAMES
    if not DATA_READ:
        DATA_READ = True
        #to read data, and return train data
        data = pd.read_table(file_name, sep=',', index_col=None)
        data = data.drop(['Qchat-10-Score', 'Case_No', 'Age_Mons', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test'], axis=1)
        COL_NAMES = list(data.columns)
        train, test = model_selection.train_test_split(data, test_size=0.25, random_state = 1)
        train, validation = model_selection.train_test_split(train, test_size= 0.1, random_state=1)
        TEST = test
        TRAIN = train
        VALID = validation

    if train_flag:
        return TRAIN, COL_NAMES
    if test_flag:
        return TEST, COL_NAMES
    if verification_flag:
        return VALID, COL_NAMES


if __name__ == '__main__':
    freeze_support()
    _, cols = read_data('../dataset/Toddler Autism dataset July 2018.csv', train_flag= True)
    cols = len(cols)
    matrix = np.zeros((cols, cols))
    for i in range(cols - 1):
        matrix[i, cols - 1] = 1

    test = DAG(matrix, name = '../dataset/Toddler Autism dataset July 2018.csv')
    print(test.matrix)
    #print(test.score())
    visited = set()
    visited.add(test)
    for i in test.neighbors():
        if i not in visited:
            print(i.matrix)
            print(i.score())
        else:
            print("already visited")

#
#
# matrix = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# print(matrix)
# gs = DAG(matrix).neighbors()
# for g in gs:
#     g = DAG(g)
#     print(sorted(g.graph.nodes()))
#     data = pd.read_csv('../dataset/test.csv', sep=',', index_col=None, header = 0)
#     for col in data.columns:
#         print(col)
#     mapping = {0: 'A1', 1: 'A2', 2: 'A3'}
#     nx.relabel_nodes(g.graph, mapping, copy = False)
#     print(sorted(g.graph))
#     model = BayesianModel(g.graph.edges)
#     print(sorted(model.nodes()))
#     model.fit(data)
#     for cpd in model.get_cpds():
#         print(cpd)


