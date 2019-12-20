#!/usr/bin/env python
# coding: utf-8

# In[144]:


import matplotlib
import pandas as pd
import sklearn
from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB #gaussian naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.neural_network import MLPClassifier

from sklearn import tree

from sklearn.metrics import accuracy_score #calculating accuracy


# In[145]:


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
    Y = y
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=size_of_test)
#     print(X_train.loc[:1])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    return X_train, X_test, Y_train, Y_test


# In[227]:


def learn_model(data,target):
    gNB = GaussianNB()
    dtc = DecisionTreeClassifier()
    lda = LinearDiscriminantAnalysis()
    rfr = RandomForestClassifier(max_depth=50, n_estimators=150, max_features=1)
    multi = MultinomialNB()
    comp = ComplementNB()
    bern = BernoulliNB()
    
    gnb_classifier = gNB.fit(data,target)
    dtc_classifier = dtc.fit(data,target)
    lda_classifier = lda.fit(data,target)
    rfr_classifier = rfr.fit(data,target)
    multi_classifier = multi.fit(data,target)
    comp_classifier = comp.fit(data,target)
    bern_classifier = bern.fit(data,target)
    
#     mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,10,10),max_iter=1000)
    MLP_classfier = mlp.fit(data, target.values.ravel())
    return gnb_classifier, dtc_classifier, lda_classifier, rfr_classifier,multi_classifier,comp_classifier,bern_classifier, MLP_classfier


# In[228]:


def classify(classifier, testdata):
    gnb_classifier = classifier[0]
    dtc_classifier = classifier[1]
    lda_classifier = classifier[2]
    rfr_classifier = classifier[3]
    multi_classifier = classifier[4]
    comp_classifier = classifier[5]
    bern_classifier = classifier[6]
    MLP_classfier = classifier[7]
    
    predicted_val_gnB = gnb_classifier.predict(testdata)
    predicted_val_dtc = dtc_classifier.predict(testdata)
    predicted_val_lda = lda_classifier.predict(testdata)
    predicted_val_rfr = rfr_classifier.predict(testdata)
    predicted_val_multi= multi_classifier.predict(testdata)
    predicted_val_comp = comp_classifier.predict(testdata)
    predicted_val_bern = bern_classifier.predict(testdata)
    predicted_val_MLP = MLP_classfier.predict(testdata)
    
    return (predicted_val_gnB, predicted_val_dtc, predicted_val_lda,predicted_val_rfr, 
            predicted_val_multi,predicted_val_comp,predicted_val_bern,predicted_val_MLP)


# In[229]:


def evaluate(actual_class, predicted_class):
    predicted_class_gnB = predicted_class[0]
    predicted_class_dtc = predicted_class[1]
    predicted_class_lda = predicted_class[2]
    predicted_class_rfr = predicted_class[3]
    predicted_class_multi = predicted_class[4]
    predicted_class_comp = predicted_class[5]
    predicted_class_bern = predicted_class[6]
    predicted_class_MLP = predicted_class[7]
    
    accuracy_gnB = accuracy_score(actual_class, predicted_class_gnB)
    accuracy_dtc = accuracy_score(actual_class, predicted_class_dtc)
    accuracy_lda = accuracy_score(actual_class, predicted_class_lda)
    accuracy_rfr = accuracy_score(actual_class, predicted_class_rfr)
    accuracy_multi = accuracy_score(actual_class, predicted_class_multi)
    accuracy_comp = accuracy_score(actual_class, predicted_class_comp)
    accuracy_bern = accuracy_score(actual_class, predicted_class_bern)
    accuracy_MLP = accuracy_score(actual_class, predicted_class_MLP)
    
    print("The accuracy score of Gaussian Naive Bayes is :",accuracy_gnB)
    print("The accuracy score of Decision Tree Classifier is :",accuracy_dtc)
    print("The accuracy score of Linear Discriminant Analysis is :",accuracy_lda)
    print("The accuracy score of Random Forest Classifier is :",accuracy_rfr)
    print("The accuracy score of Multinomial Naive Bayes is :",accuracy_multi)
    print("The accuracy score of Complement Naive Bayes is :",accuracy_comp)
    print("The accuracy score of Bernoulli Naive Bayes is :",accuracy_bern)
    print("The accuracy score of ANN MLP is :",accuracy_MLP)


# In[230]:


print("preprocessing data.....")
data = preprocess('../dataset/Toddler Autism dataset July 2018.csv',0.2)
trainingX = data[0]
testX =  data[1]
trainingY = data[2]
testY = data[3]


# In[231]:


print("Learning model.....")
model = learn_model(trainingX,trainingY)


# In[232]:


print("Classifying test data......")      
predictedY = classify(model, testX)


# In[233]:


print("Evaluating results.....")
evaluate(testY,predictedY)


# In[ ]:





# In[ ]:




