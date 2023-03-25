import cv2
import numpy as np
import face_recognition

img_bgr = face_recognition.load_image_file('sk.jpeg')
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

face = face_recognition.face_locations(img_rgb)[0]
copy = img_rgb.copy()

cv2.rectangle(copy, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
cv2.imshow('copy', copy)
cv2.waitKey(300)
