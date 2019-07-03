
""" 	prediction on GISETTE data
	print the accuracy score of the model on GISETTE trainning dataset
	export the prediction of the model on GISITTE test dataset
"""

data_dir = 'GISETTE'             
data_name = 'gisette'
#!wc $data_dir/* 


# pandas module importation
import pandas as pd
#gisette_train.data importation as a dataframe
#removal of NaN values
trainData = pd.read_csv(data_dir  + '/' + data_name+"_train.data", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
trainData = trainData.dropna(axis=1, how="any")
#trainData # the standard output dataframe


#gisette_train.labels importation as a dataframe
#removal of NaN values
trainLabels = pd.read_csv(data_dir  + '/' + data_name+"_train.labels", sep=" ",header=None, encoding='utf8') 
trainLabels = trainLabels.dropna(axis=1, how="any")
#trainLabels # the standard output dataframe


#gisette_valid.data importation as a dataframe
#removal of NaN values
validData = pd.read_csv(data_dir  + '/' + data_name+"_valid.data", sep=" ",header=None, encoding='utf8')
validData=validData.dropna(axis=1,how="any")
#validData  # the standard output dataframe


#gisettee_valid.labels importation as a dataframe
#removal of NaN values
validLabels = pd.read_csv(data_name+"_valid.labels", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
validLabels=validLabels.dropna(axis=1,how="any")
#validLabels  # the standard output dataframe


#gisette_valid.data importation as a dataframe
#removal of NaN values
testData = pd.read_csv(data_dir  + '/' + data_name+"_test.data", sep=" ",header=None, encoding='utf8')
testData=testData.dropna(axis=1,how="any")


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.feature_selection import SelectPercentile


select = SelectPercentile(percentile = 50)
select.fit(trainData,trainLabels)

selectedTrainData = select.transform(trainData)
selectedTestData =select.transform(testData) 

#10-fols cross-validation with KNeighborsClassifier
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
clf.fit(selectedTrainData, trainLabels)
labelsPred = clf.predict(selectedTestData);
pd.DataFrame(labelsPred).to_csv("../gisette_test.predict", index=False,header=False)
print (cross_val_score(clf,selectedTrainData,trainLabels,cv=10, scoring="accuracy").mean())

