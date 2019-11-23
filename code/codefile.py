import pandas as pd
import sklearn
from sklearn import model_selection


def preprocess(file,size_of_test):
    # read the csv
    data = pd.read_table(file, sep=',', index_col=None)

#     print('Shape of DataFrame: {}'.format(data.shape))
    
    # drop unwanted columns
    data = data.drop(['Qchat-10-Score','Ethnicity'], axis=1)

    # create X and Y datasets for training
    x = data.drop(['Class/ASD Traits '],1)
    y = data['Class/ASD Traits ']
    
    # convert the data to categorical values
    X = pd.get_dummies(x)
    Y = pd.get_dummies(y)
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=size_of_test)

#     print(X_train.loc[:1])
#     print(X_train.shape)
#     print(X_test.shape)
#     print(Y_train.shape)
#     print(Y_test.shape)
preprocess('../dataset/Toddler Autism dataset July 2018.csv',0.2)
