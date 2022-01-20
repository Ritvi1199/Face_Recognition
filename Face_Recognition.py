 !pip install face_recognition
     !mkdir known
     !mkdir Unknown

     
import face_recognition
import numpy as np
import cv2
import os
from google.colab.patches import cv2_imshow

def read_img(path):
  img =cv2.imread(path)
  (h, w) = img.shape[:2]
  width = 500
  ratio = width / float(w)
  height =int(h * ratio)
  return cv2.resize(img, (width,height))

  
known_encodings = []
known_name =[]
known_dir = '/content/known'

for file in os.listdir(known_dir):
  img = read_img(known_dir + '/' + file)
  img_enc = face_recognition.face_encodings(img)[0]
  known_encodings.append(img_enc)
  known_name.append(file.split('.')[0])

unknown_dir = '/content/Unknown'
for file in os.listdir(unknown_dir):
  print("Processing", file)
  img = read_img(unknown_dir + '/' + file)
  img_enc = face_recognition.face_encodings(img)[0]

  result = face_recognition.compare_faces(known_encodings, img_enc)

  for i in range(len(result)):
    if result[i]:
      name = known_name[i]
      (top, right, bottom, left) = face_recognition.face_locations(img)[0]
      cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
      cv2.putText(img, name, (left+2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 2, (128, 0, 0), 1)
      cv2_imshow(img)

  print(result)
