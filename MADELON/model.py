''' executing this fails print the accuracy of decision tre on madelon_valid and
export the prediction on madelon_test.data in madelon_test.predict'''

# diractories variable setting

data_dir = 'MADELON'             
data_name = 'madelon'
#!wc $data_dir/* 

# pandas module importation
import pandas as pd
#madelon_valid_train.data importation as a dataframe
#removal of NaN values
trainData = pd.read_csv(data_dir  + '/' + data_name+"_train.data", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
trainData = trainData.dropna(axis=1, how="any")
#trainData # the standard output dataframe


#madelon_valid_train.labels importation as a dataframe
#removal of NaN values
trainLabels = pd.read_csv(data_dir  + '/' + data_name+"_train.labels", sep=" ",header=None, encoding='utf8') 
trainLabels = trainLabels.dropna(axis=1, how="any")
#trainLabels # the standard output dataframe

#madelon_valid_valid.data importation as a dataframe
#removal of NaN values
validData = pd.read_csv(data_dir  + '/' + data_name+"_valid.data", sep=" ",header=None, encoding='utf8')
validData=validData.dropna(axis=1,how="any")
#validData  # the standard output dataframe

#madelon_valid_valid.data importation as a dataframe
#removal of NaN values
validLabels = pd.read_csv(data_name+"_valid.labels", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
validLabels=validLabels.dropna(axis=1,how="any")
#validLabels  # the standard output dataframe

#madelon_valid.data importation as a dataframe
#removal of NaN values
testData = pd.read_csv(data_dir  + '/' + data_name+"_test.data", sep=" ",header=None, encoding='utf8')
testData=testData.dropna(axis=1,how="any")
#testData  # the standard output dataframe

from sklearn import tree
#from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(trainData, trainLabels)
labelsPred = clf.predict(testData);
#acc = accuracy_score(validLabels,labelsPred)
#acc

#export the data 
pd.DataFrame(labelsPred).to_csv("../madelon_test.predict", index=False)

#10-fols cross-validation with DecisionTree
#from sklearn import tree
from sklearn.model_selection import cross_val_score
clf = tree.DecisionTreeClassifier()
print (cross_val_score(clf,validData,validLabels,cv=10, scoring="accuracy").mean())
