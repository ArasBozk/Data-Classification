import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

names = ['CodeNum','ClumpThickness', 'USize', 'UShape','Margin','SSize', 'BrNuc','BlandCh', 'Normal', 'Mithoses', 'Class']

f = open("breast-cancer-wisconsin.data","r")
contents = [names]
counter = 0

####MANUPILATION
for line in f:     
    contents.append(list(map(int, line[:-1].replace("?", "-1").split(","))))                       
    counter += 1
    
f.close()
f = open("breast-cancer-wisconsin.data.txt","w")

print(counter)

length = len(names)
for i in range(0, counter):
    for j in range (0,length):
        f.write(str(contents[i][j]))
        if(j != length-1):
            f.write(",")
    f.write("\n")
f.close()
####MANUPILATION
    
df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999 , inplace=True)
df.drop(['CodeNum'],1,inplace=True)

x= np.array(df.drop(['Class'],1))
y= np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

accuracy_l = [[],[]]

for i in range(1,558):
    clf= neighbors.KNeighborsClassifier(i)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test,y_test)
    print("k: ", i, "accuracy:", accuracy)
    accuracy_l[0].append(i)
    accuracy_l[1].append(accuracy)

#EXAMPLE                                                                                                                                                                                                    
example= np.array([1,2,3,4,1,3,2,1,2])
example= example.reshape(1,-1)
prediction = clf.predict(example)
print(prediction)

#EXAMPLE
    
plt.plot(accuracy_l[0], accuracy_l[1])
plt.show()


      
        

    
