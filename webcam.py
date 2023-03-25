import numpy as np
import cv2
import os

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) 
cap.set(4,480) 
face_id = input('\n Enter user id and press Enter ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count =0
while(True):
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        count+=1
        cv2.imwrite("dataset/User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
    cv2.imshow('video',frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
    elif count >= 10:
         break
    

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()