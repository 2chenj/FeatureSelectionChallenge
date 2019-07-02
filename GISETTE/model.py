# sumery on GISETTE DATA
''' by execution this file u get the accuracy of Nearest Neighbour on
gisette_valid  by cross validation 
and export the accuracy on test_data in gesitte_test.predict'''

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

#testData  # the standard output dataframe


#export the data 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
clf.fit(trainData, trainLabels)
labelsPred = clf.predict(testData);
pd.DataFrame(labelsPred).to_csv("../gisette_test.predict", index=False)

#10-fols cross-validation with DecisionTree
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
print (cross_val_score(clf,validData,validLabels,cv=10, scoring="accuracy").mean())
