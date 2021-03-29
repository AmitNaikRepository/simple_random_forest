from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import json 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

#lets read the data 

x_train=np.genfromtxt('data/train_features.csv')
y_train=np.genfromtxt('data/train_labels.csv')
x_test=np.genfromtxt('data/test_features.csv')
y_test=np.genfromtxt('data/test_labels.csv')

#fit the model 

depth=2
clf=RandomForestClassifier(max_depth=depth)
clf.fit(x_train,y_train)


acc=clf.score(x_test,y_test)

print(acc)

with open('metrics.txt','w') as outfile:
    outfile.write('Accuracy :' + str(acc)+'\n')


#plot it 
disp=plot_confusion_matrix(clf,x_test,y_test,normalize='true',cmap=plt.cm.Blues)
plt.savefig('confusion_matrics.png')