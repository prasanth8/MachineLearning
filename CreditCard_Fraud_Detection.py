import numpy as np
import pandas as pn
import seaborn as sb
import matplotlib.pyplot as plt

data=pn.read_csv('creditcard.csv')
data.columns
print(data.shape)
data.head()
data.describe()

class_name={0:"Not Fraud",1:"Fruad"}
data.Class.value_counts().rename(index=class_name)

from sklearn.cross_validation import train_test_split

feature_name=data.iloc[:,1:30].columns
target=data.iloc[:1,30:].columns
print(feature_name)
print(target)

data_feature=data[feature_name]
data_target=data[target]
X_train,X_test,y_train,y_test=train_test_split(data_feature,data_target,train_size=0.70,test_size=0.30,random_state=1)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
from sklearn.metrics import confusion_matrix

lr.fit(X_train,y_train.values.ravel())

Test_data=lr.predict(X_test)

final_data=confusion_matrix(y_test,Test_data)
final_data