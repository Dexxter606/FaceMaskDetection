import cv2
import datetime
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('mymodel.h5')
img_width, img_height = 150,150

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_count_full = 0
# parameters for text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1, 1)
fontScale = 1
color = (255, 0, 0)
thickness = 2

while cap.isOpened():
    img_count_full += 1
    # read image from webcam
    response, color_img = cap.read()

    if response == False:
        print('No face detected')
        break
    # resize image with 50% ratio


    # convert to grayscale
    gray_img = cv2.cvtColor(color_img , cv2.COLOR_BGR2GRAY)

    # Detect the face
    faces = face_cascade.detectMultiScale(gray_img, 1.3 , 4)

    # take face the predict class mask or not mask then draw rectangle
    img_count = 0
    for (x,y,w,h) in faces:
        org = (x-20 , y-20)
        img_count += 1
        color_face = color_img[y:y+h , x:x+w]
        cv2.imwrite('faces/input/%d%dface.jpg'%(img_count_full, img_count),color_face)
        img = image.load_img('faces/input/%d%dface.jpg'%(img_count_full, img_count),target_size= (img_width,img_height))

        img = image.img_to_array(img)/255
        img = np.expand_dims(img,axis=0)
        pred_prob = model.predict(img)
        pred = np.argmax(pred_prob)

        if pred == 0:
            print("User with mask -predic= ", pred_prob[0][0])
            class_label = 'Mask'
            color = (255,0,0)
            cv2.imwrite('faces/with_mask/%d%dface.jpg'%(img_count_full,img_count),color_face)
        else:
            print('user not wearing mask - prob =' , pred_prob[0][1])
            class_label = "No Mask"
            color =(0,255,0)
            cv2.imwrite('faces/without_mask/%d%dface.jpg'%(img_count_full,img_count),color_face)
        cv2.rectangle(color_img, (x,y),(x+w, y+h), (0, 0, 255),3)
        cv2.putText(color_img,class_label ,org, font, fontScale, color , thickness, cv2.LINE_AA)
    cv2.imshow('LIVE face mask detection', color_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()