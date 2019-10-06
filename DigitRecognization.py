from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import cv2
import matplotlib.pyplot as plt


minist=datasets.load_digits()
X_train,X_test,y_train,y_test=train_test_split(minist.data,minist.target,test_size=0.25,random_state=42)

X_train,val_train,y_train,val_test=train_test_split(minist.data,minist.target,test_size=0.10,random_state=84)
kvals=range(1,30,2)
acc=[]
for value in kvals:
    model=KNeighborsClassifier(n_neighbors=value)
    model.fit(X_train,y_train)
    score=model.score(val_train,val_test)
    print("k=%d,Accuirecy=%.2f%%"%(value,score*100))
    acc.append(score)
max_value=np.argmax(acc)
kvals[max_value]
model=KNeighborsClassifier(n_neighbors=kvals[max_value])
model.fit(X_train,y_train)
prediction=model.predict(X_test)

for i in np.random.randint(0,high=len(y_test),size=(5)):
    image=X_test[i]
    print([image][0])
    prediction=model.predict([image])[0]
    plt.imshow(np.array(image,dtype='float').reshape((8,8)),cmap="gray")
    plt.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
    print("I think the digit is ={}".format(prediction))
    plt.show()
    cv2.waitKey(0)
image=cv2.imread('nine.jpg',0)

print(image.shape)
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()