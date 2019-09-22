import cv2
import numpy as np

import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)

faceCascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

ret,frame=cap.read()


#to store face data
faceData=[]
faceCount=0

while True:
    
    ret,frame=cap.read()
    if ret == True:
        grayFace=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(grayFace,1.3,5)

        for(x,y,w,h) in faces:
            cropedFace=frame[y:y+h,x:x+h:]
            resizedFace =cv2.resize(cropedFace,(50,50))
            faceData.append(resizedFace)
           # cv2.imwrite('sample'+str(facecount)+'jpg',resizedFace)
            faceCount+=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow('capturing Frames',frame)
        if cv2.waitKey(1)==27 or len(faceData) >=100:
            break 
    else:
        print("cameraerror")
cap.release()
cv2.destroyAllWindows()

faceData=np.asarray(faceData)

np.save('Name1',faceData)