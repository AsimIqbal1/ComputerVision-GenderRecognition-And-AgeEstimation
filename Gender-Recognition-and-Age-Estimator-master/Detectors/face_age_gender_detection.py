#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pafy
import PIL
from PIL import Image
from PIL import ImageChops
from PIL import ImageStat

def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('haar_cascade_frontal_face/deploy_age.prototxt', 'haar_cascade_frontal_face/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('haar_cascade_frontal_face/deploy_gender.prototxt', 'haar_cascade_frontal_face/gender_net.caffemodel')
    return (age_net, gender_net)




# url of the video to predict Age and gender



url = 'https://www.youtube.com/watch?v=c07IsbSNqfI&feature=youtu.be'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype='mp4')
cap = cv2.VideoCapture(0)

cap.set(3, 480)  # set width of the frame
cap.set(4, 640)  # set height of the frame
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = [
    '(0, 2)',
    '(4, 6)',
    '(8, 12)',
    '(15, 20)',
    '(25, 32)',
    '(38, 43)',
    '(48, 53)',
    '(60, 100)'
    ]
gender_list = ['Male', 'Female']

diff_img_file = "file.png"

def diff(im1_file, 
         im2_file, 
         delete_diff_file=False, 
         diff_img_file=diff_img_file):
    im1 = Image.open(im1_file)
    im2 = Image.open(im2_file)
    diff_img = ImageChops.difference(im1,im2)
    diff_img.convert('RGB').save(diff_img_file)
    stat = ImageStat.Stat(diff_img)
    # can be [r,g,b] or [r,g,b,a]
    sum_channel_values = sum(stat.mean)
    max_all_channels = len(stat.mean) * 100
    diff_ratio = sum_channel_values/max_all_channels
    if delete_diff_file:
        remove(diff_img_file)
    return diff_ratio

def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    i=0
    while True:
        (ret, image) = cap.read()
        face_cascade = cv2.CascadeClassifier('F:\VS Code\Projects\Gender-Recognition-and-Age-Estimator-master\haar_cascade_frontal_face\haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0xFF, 0xFF, 0), 2)
                frame = image[y: y+h, x: x+w]
                img_name = "fold/face"+str(i)+".png"
                if not os.path.exists("fold"):
                    os.makedirs("fold")
                cv2.imwrite(img_name, frame)
                i=i+1

    # Get Face

                face_img = image[y:y + h, h:h + w].copy()
                blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227),MODEL_MEAN_VALUES, swapRB=False)

    # Predict Gender

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]
                print("Gender : " + gender)

            # Predict Age

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                print ("Age Range: " + age)
                overlay_text = '%s %s' % (gender, age)
                cv2.putText(image, overlay_text,(x, y), font, 1, (0xFF, 0xFF, 0xFF), 2, cv2.LINE_AA)
        cv2.imshow('frame', image)

    # 0xFF is a hexadecimal constant which is 11111111 in binary.

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    im1 = "F:\\VS Code\\Projects\\Gender-Recognition-and-Age-Estimator-master\\fold\\face0.png"
    im2 = "F:\\VS Code\\Projects\\Gender-Recognition-and-Age-Estimator-master\\fold\\face20.png"

    print(str(diff(im1,im2)))
    



if __name__ == '__main__':
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)