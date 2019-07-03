""" prediction on MADELON data

	prints the score of the model on MADELON training Set with closs validation
	export the model prediction on MADELON test set
"""



##IMPORTATION DES DONNEES


# diractories variable setting
data_dir = 'MADELON'             
data_name = 'madelon'
# pandas module importation

import pandas as pd
#madelon_valid_train.data importation as a dataframe
trainData = pd.read_csv(data_dir  + '/' + data_name+"_train.data", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
trainData = trainData.dropna(axis=1, how="any") #removal of NaN values

trainLabels = pd.read_csv(data_dir  + '/' + data_name+"_train.labels", sep=" ",header=None, encoding='utf8') 
trainLabels = trainLabels.dropna(axis=1, how="any")

validData = pd.read_csv(data_dir  + '/' + data_name+"_valid.data", sep=" ",header=None, encoding='utf8')
validData=validData.dropna(axis=1,how="any")

validLabels = pd.read_csv(data_name+"_valid.labels", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
validLabels=validLabels.dropna(axis=1,how="any")

testData = pd.read_csv(data_dir  + '/' + data_name+"_test.data", sep=" ",header=None, encoding='utf8')
testData=testData.dropna(axis=1,how="any")

from sklearn import tree
import numpy as np
from sklearn.feature_selection import SelectPercentile

# FEATURES SELECTION
select = SelectPercentile(percentile = 65)
select.fit(trainData,trainLabels)

selectedTrainData = select.transform(trainData)
selectedTestData =select.transform(testData) 

#PREDICTION AND EXPORT

#10-fols cross-validation with DecisionTree
from sklearn.model_selection import cross_val_score
clf =tree.DecisionTreeClassifier()
clf.fit(selectedTrainData, trainLabels)
labelsPred = clf.predict(selectedTestData);
pd.DataFrame(labelsPred).to_csv("../madelon_test.predict", index=False , header=False) #exportation of labelsPred
print (cross_val_score(clf,selectedTrainData,trainLabels,cv=10, scoring="accuracy").mean())


