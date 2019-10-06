import seaborn as sb
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt

iris=pn.read_csv('iris.csv')
iris.head()
iris.set_index('Id',inplace=True)

iris.head()
iris_feature=iris.iloc[:,1:len(iris.columns)-1].columns
iris_target=iris.iloc[:1,len(iris.columns)-1:].columns
#iris_target=iris.iloc[len(iris.columns)].columns


#iris_target
	
iris_fea=iris[iris_feature]
iris_tar=iris[iris_target]	

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(iris_fea,iris_tar,test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
LogReg=LogisticRegression()
LogReg.fit(X_train,y_train.values.ravel())


Pred_data=LogReg.predict(X_test)

confusion_matrix(y_test,Pred_data)
generateClassificationReport(y_test,Pred_data)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def generateClassificationReport(y_test,pred):
    print(classification_report(y_test,pred))
    print(confusion_matrix(y_test,pred))
    print('{0:.16f}'.format(accuracy_score(y_test,pred)))

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train.values.ravel())

svm_pred=classifier.predict(X_test)
generateClassificationReport(y_test,svm_pred)