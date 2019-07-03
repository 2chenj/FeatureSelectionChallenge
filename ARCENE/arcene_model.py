
""" 	prediction on MADELON data
	print the occuracy score  of the model on ARCENE trainng dataset 
	export the model prediction on ARCENE test dataset
 """




#IMPORTATION OF DATASETS

# diretories variable setting
data_dir = 'ARCENE'             
data_name = 'arcene'
#!wc $data_dir/* 


# pandas module importation
import pandas as pd


#arcene_train.data importation as a dataframe

trainData = pd.read_csv(data_dir  + '/' + data_name+"_train.data", sep=" ", header=None)  # The data 
trainData = trainData.dropna(axis=1, how="any") #removal of NaN values

trainLabels = pd.read_csv(data_dir  + '/' + data_name+"_train.labels", sep=" ",header=None, encoding='utf8') 
trainLabels = trainLabels.dropna(axis=1, how="any")

validData = pd.read_csv(data_dir  + '/' + data_name+"_valid.data", sep=" ",header=None, encoding='utf8')
validData=validData.dropna(axis=1,how="any")

validLabels = pd.read_csv(data_name+"_valid.labels", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
validLabels=validLabels.dropna(axis=1,how="any")

testData = pd.read_csv(data_dir  + '/' + data_name+"_test.data", sep=" ",header=None, encoding='utf8')  # The data are loaded as a Pandas Data Frame
testData=testData.dropna(axis=1,how="any")

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_selection import SelectPercentile

#FEATURES SELECTION
select = SelectPercentile(percentile = 50)
select.fit(trainData,trainLabels)

selectedTrainData = select.transform(trainData)
selectedTestData =select.transform(testData) 

#PREDICTION AND EXPORTS
#10-fols cross-validation with KNeighborsClassifier
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
clf.fit(selectedTrainData, trainLabels)
labelsPred = clf.predict(selectedTestData);
pd.DataFrame(labelsPred).to_csv("../arcene_test.predict", index=False,header=False)
print (cross_val_score(clf,selectedTrainData,trainLabels,cv=10, scoring="accuracy").mean())


