#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import sklearn
from sklearn import model_selection
from IPython.display import Image  
from sklearn import tree
import pydotplus
from sklearn.naive_bayes import GaussianNB #gaussian naive Bayes classifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree.export import export_text
from pyswip imp ort Prolog
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score #calculating accuracy


# In[21]:



def preprocess(file,size_of_test):
    # read the csv
    data = pd.read_table(file, sep=',', index_col=None

#     print('Shape of DataFrame: {}'.format(data.shape))
    
    
    # drop unwanted columns
    data = data.drop(['Qchat-10-Score','Ethnicity', 'Case_No'], axis=1)

    # create X and Y datasets for training
    x = data.drop(['Class/ASD Traits '],1)
    y = data['Class/ASD Traits ']
    
    # convert the data to categorical values)
    X = pd.get_dummies(x)
    Y = y
    column_names = list(X.columns) 
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=size_of_test, random_state = 1)
#     print(X_train.loc[:1])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    return X_train, X_test, Y_train, Y_test, column_names


# In[22]:


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
    return gnb_classifier, dtc_classifier, lda_classifier, rfr_classifier,multi_classifier,comp_classifier,bern_classifier


# In[23]:


def classify(classifier, testdata):
    gnb_classifier = classifier[0]
    dtc_classifier = classifier[1]
    lda_classifier = classifier[2]
    rfr_classifier = classifier[3]
    multi_classifier = classifier[4]
    comp_classifier = classifier[5]
    bern_classifier = classifier[6]
    
    predicted_val_gnB = gnb_classifier.predict(testdata)
    predicted_val_dtc = dtc_classifier.predict(testdata)
    predicted_val_lda = lda_classifier.predict(testdata)
    predicted_val_rfr = rfr_classifier.predict(testdata)
    predicted_val_multi= multi_classifier.predict(testdata)
    predicted_val_comp = comp_classifier.predict(testdata)
    predicted_val_bern = bern_classifier.predict(testdata)
    
    return predicted_val_gnB, predicted_val_dtc, predicted_val_lda, predicted_val_rfr


# In[24]:


def evaluate(actual_class, predicted_class):
    predicted_class_gnB = predicted_class[0]
    predicted_class_dtc = predicted_class[1]
    predicted_class_lda = predicted_class[2]
    predicted_class_rfr = predicted_class[3]
    
    accuracy_gnB = accuracy_score(actual_class, predicted_class_gnB)
    accuracy_dtc = accuracy_score(actual_class, predicted_class_dtc)
    accuracy_lda = accuracy_score(actual_class, predicted_class_lda)
    accuracy_rfr = accuracy_score(actual_class, predicted_class_rfr)
    
    print("The accuracy score of Gaussian Naive Bayes is :",accuracy_gnB)
    print("The accuracy score of Decision Tree Classifier is :",accuracy_dtc)
    print("The accuracy score of Linear Discriminant Analysis is :",accuracy_lda)
    print("The accuracy score of Random Forest Classifier is :",accuracy_rfr)


# In[25]:


def DTS(data,target,names_of_features):
    dt = DecisionTreeClassifier(max_depth = 5)

    # Train model
    d_tree= dt.fit(data, target)
    dot_data = tree.export_graphviz(dt,feature_names = names_of_features, class_names = ["No", "Yes"])
    #graph = pydotplus.graph_from_dot_data(dot_data)
    #Image(graph.create_png())
    #graph.write_png("dtree.png")
    
    
    #r = export_text(d_tree, feature_names=names_of_features)
    
    #f = open("dts_rules.txt", "w")
    #f.write(r)
    #f.close()


# In[26]:


print("preprocessing data.....")
data = preprocess('../dataset/Toddler Autism dataset July 2018.csv',0.2)
trainingX = data[0]
testX =  data[1]
trainingY = data[2]
testY = data[3]
feature_names = data[4]
print(len(feature_names))


# In[27]:


print("making DTS...")
DecisonTree = DTS(trainingX,trainingY,feature_names)


# In[11]:


print("Learning model.....")
model = learn_model(trainingX,trainingY)


# In[12]:


print("Classifying test data......")      
predictedY = classify(model, testX)


# In[13]:


print("Evaluating results.....")
evaluate(testY,predictedY)

print("Querying")
prolog = Prolog()
prolog.consult("prolog_rules.pl")

predicted = []
for index, data in testX.iterrows():
    to_query = (data['A1'], data['A2'], data['A3'], data['A4'], data['A5'], data['A6'], data['A7'], data['A8'], data['A9'], data['A10'], data['Age_Mons'])
    query_str = "autistic(" + (",".join([str(i) for i in to_query])) + ")."
    print(query_str)
    if (bool(list((prolog.query(query_str))))):
        predicted.append("Yes")
    else:
        predicted.append("No")

print(accuracy_score(testY, predicted))
print(confusion_matrix(testY, predicted))
print(bool(list((prolog.query("autistic(0,0,0,0,0,0,0,0,0,0,0).")))))
print(bool(list(prolog.query("autistic(0,0,0,1,1,0,0,0,0,0,0)."))))
print(bool(list(prolog.query("autistic(0,0,0,1,0,0,0,0,0,0,0)."))))
# In[ ]:





# In[ ]:




