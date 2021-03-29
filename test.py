from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
import numpy as np 
import os as os

seed=42
x,y=make_classification(n_samples=1000,random_state=seed)


#make a prediction for the make dataset and the prediction 


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=seed)

#lets take an exaple if we have the data dir 

if not os.path.exists('data'):
    os.mkdir('data')

np.savetxt('data/train_feature.csv',x_train)
np.savetxt('data/test_feature.csv',x_test)
np.savetxt('data/train_labels.csv',y_train)
np.savetxt('data/test_labels.csv',y_test)


