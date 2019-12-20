from DAG import *
from pomegranate import *
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_table('../dataset/Toddler Autism dataset July 2018.csv', sep=',', index_col=None)
data = data.drop(['Qchat-10-Score', 'Case_No', 'Age_Mons', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test'], axis=1)
COL_NAMES = list(data.columns)
train, test = model_selection.train_test_split(data, test_size=0.25, random_state = 1)
print("going to learn")
model = BayesianNetwork.from_samples(train, algorithm='exact', state_names = COL_NAMES)
model.plot()
final_predictions_actual = []
final_predictions_model = []
for row in test.iterrows():
    to_pred = []
    label = random.random() < 0.5 #including the label with RHS
    if label:
        to_pred.append(len(COL_NAMES)-1)
        final_predictions_actual.append(row[1].values[-1])
    variables_to_hide = int(round(np.random.normal(2),0))
    for i in range(variables_to_hide):
        column = random.randint(0, len(COL_NAMES) - 2)
        to_pred.append(column)
        final_predictions_actual.append(row[1].values[column])
    if to_pred:
        for i in to_pred:
            row[1].values[i] = None
        prediction = model.predict([row[1].values])
        print(prediction, to_pred)
        for i in to_pred:
            print(i)
            final_predictions_model.append(prediction[0][i])
print(accuracy_score(final_predictions_actual, final_predictions_model))
    
    
