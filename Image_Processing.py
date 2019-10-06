import cv2
import numpy as np
from operator import eq
import matplotlib.pyplot as plt
count=0
flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
img=cv2.imread('Prasanth.jpg',0)

cv2.namedWindow('Arun',cv2.WINDOW_NORMAL)
cv2.imshow('Arun',img)
k=cv2.waitKey(0)
if eq(chr(k),'s'):
    cv2.imwrite('Changed.png',img)
    cv2.destroyAllWindows()
    print('Successfully Saved')
else :
    cv2.destroyAllWindows()
    print('Successfully Closed')

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

image=cv2.imread('photo_2.jpg',1)
canny=cv2.Canny(image,100,200)
canny2=cv2.Canny(image,200,200)
canny3=cv2.Canny(image,100,300)
cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge',cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge2',cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge3',cv2.WINDOW_NORMAL)
cv2.imshow('Original',image)
cv2.imshow('Edge',canny)
cv2.imshow('Edge2',canny2)
cv2.imshow('Edge3',canny3)
k=cv2.waitKey(0)
if eq(chr(k),'s'):
    cv2.destroyAllWindows()


