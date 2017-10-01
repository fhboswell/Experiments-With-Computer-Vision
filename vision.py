import sys

import numpy as np
import cv2

img = cv2.imread('pitrain.png', 0)

#################      Now finding Contours         ###################

#img = cv2.imread('star.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print( M )

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)


samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        
        if  h>28:
            cv2.rectangle(im2,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im2)
            key = cv2.waitKey(0)
     

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
