import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

#Decision tree

#use this classification algorithm to build a model from the historical data of patients,
#and their response to different medications. Then you will use the trained decision tree to predict the class of an unknown patient,
#or to find a proper drug for a new patient.

#can use the training part of the dataset to build a decision tree, 
#and then use it to predict the class of an unknown patient, or to prescribe a drug to a new patient.


#load data
my_data=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv', delimiter=",")
my_data.head()

#size of data
my_data.shape

#Preprocessing

#Removing the column containing the target name since it doesn't contain numeric values.
x=my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
x[0:5]


#We can still convert these features to numerical values using the LabelEncoder() method
#to convert the categorical variable into dummy/indicator variables.

from sklearn import preprocessing
le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1]=le_sex.transform(x[:,1])

le_BP=preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
x[:,2]=le_BP.transform(x[:,2])

le_chol=preprocessing.LabelEncoder()
le_chol.fit(['NORMAL','HIGH'])
x[:,3]=le_chol.transform(x[:,3])

x[0:5]


#Target variable
y=my_data['Drug']
y[0:5]


#setting up decision tree
from sklearn.model_selection import train_test_split
#test_size represents the ratio of the testing dataset, 
#and the random_state ensures that we obtain the same splits.
x_trainset,x_testset,y_trainset,y_testset=train_test_split(x,y,test_size=0.3,random_state=3)
print('Shape of x training set {}'.format(x_trainset.shape),'&',' Size of y training set {}'.format(y_trainset.shape))
print('Shape of X training set {}'.format(x_testset.shape),'&',' Size of Y training set {}'.format(y_testset.shape))


#modelling

#create an instance of the DecisionTreeClassifier called drugTree
#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#fit the data
drugTree.fit(x_trainset,y_trainset)

#Prediction
predTree=drugTree.predict(x_testset)
print(predTree[0:5])
print(y_testset[0:5])

#Evaluation

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTree's Accuracy: ",metrics.accuracy_score(y_testset,predTree))
#Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
#In multilabel classification, the function returns the subset accuracy. 
#If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.


#Visualization of the tree
from sklearn.tree import export_graphviz
import subprocess
export_graphviz(drugTree, out_file='tree.dot', filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
# Convert the dot file to a PNG image using the Graphviz command line tool
subprocess.call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])