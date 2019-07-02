
''' the execution of this print the accuracy of the model Nearest Neigbour
by cross validation on arcenne_test.data

'''





# diretories variable setting
data_dir = 'ARCENE'             
data_name = 'arcene'
#!wc $data_dir/* 


# pandas module importation
import pandas as pd


#arcene_train.data importation as a dataframe
#removal of NaN values
trainData = pd.read_csv(data_dir  + '/' + data_name+"_train.data", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
trainData = trainData.dropna(axis=1, how="any")
#trainData # the standard output dataframe


#arcene_train.labels importation as a dataframe
#removal of NaN values
trainLabels = pd.read_csv(data_dir  + '/' + data_name+"_train.labels", sep=" ",header=None, encoding='utf8') 
trainLabels = trainLabels.dropna(axis=1, how="any")
#trainLabels # the standard output dataframe


#arcene_valid.data importation as a dataframe
#removal of NaN values
validData = pd.read_csv(data_dir  + '/' + data_name+"_valid.data", sep=" ",header=None, encoding='utf8')
validData=validData.dropna(axis=1,how="any")
#validData  # the standard output dataframe


#arcene_valid.data importation as a dataframe
#removal of NaN values
validLabels = pd.read_csv(data_name+"_valid.labels", sep=" ", header=None)  # The data are loaded as a Pandas Data Frame
validLabels=validLabels.dropna(axis=1,how="any")
#validLabels  # the standard output dataframe


#arcene_test.data importation as a dataframe
#removal of NaN values
testData = pd.read_csv(data_dir  + '/' + data_name+"_test.data", sep=" ",header=None, encoding='utf8')  # The data are loaded as a Pandas Data Frame
testData=testData.dropna(axis=1,how="any")
#vtestData  # the standard output dataframe



#10-fols cross-validation with DecisionTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
clf = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
print (cross_val_score(clf,trainData,trainLabels,cv=10, scoring="accuracy").mean())
