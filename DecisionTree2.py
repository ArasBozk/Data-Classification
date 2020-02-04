import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn import preprocessing
import seaborn as sns
from sklearn import model_selection
import graphviz
from sklearn.model_selection import train_test_split

df = pd.read_csv("bank-full.csv")
df.head()
print(df) 

target = df['deposit']
del df['deposit']
del df['contact']
del df['month']
del df['day']

le = preprocessing.LabelEncoder() 

for column in df:
    df[column]=le.fit_transform(df[column])

x_train, x_test, y_train, y_test = model_selection.train_test_split(df, target, test_size=0.2, random_state=0)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


arrFm = ["age","job","martial","education","default","balance","housing","loan","duration","campaign","pdays","previous","poutcome"]

for i in range (1,10):
    dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=i)
    dt2.fit(x_train, y_train)
    dt2_score_train = dt2.score(x_train, y_train)
    print("For Depth: ", i)
    print("Training score: ",dt2_score_train)
    dt2_score_test = dt2.score(x_test, y_test)
    print("Testing score: ",dt2_score_test)
    viz_tree = tree.export_graphviz(dt2, out_file=None , feature_names=arrFm , filled=True, rounded=True)
    graph =graphviz.Source(viz_tree)
    graph.render("duration_tree_"+str(i))
    
dt111 = tree.DecisionTreeClassifier()
dt111.fit(x_train, y_train)
dt111_score_train = dt111.score(x_train, y_train)
print("Depth:  Max")
print("Training score: ",dt111_score_train)
dt111_score_test = dt111.score(x_test, y_test)
print("Testing score: ",dt111_score_test)

