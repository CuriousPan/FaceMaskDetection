import cv2
import numpy as np
import os
import tensorflow as tf
import keras
import pickle

labels = ["with_mask", "without_mask"]

#path to your model
os.chdir("C:\PythonStuff\MasksProject\model")
model = keras.models.load_model("model.h5")

#path to haar cascade
os.chdir("C:\PythonStuff\MasksProject")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0) # 0 for turning on the WebCam

while True:
    
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:        
        toTest = gray[y:y+h, x:x+w]
        toTest = cv2.resize(toTest, (100, 100))
        #cv2.imshow("test2", toTest) shows the cropped image which is sent to cnn (optional)
        toTest = toTest.reshape(1, 100, 100, 1)
        
        prediction = model.predict(toTest, verbose=0)
        
        if np.argmax(prediction) == 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (48, 216, 48), 2)
            cv2.putText(img, "Mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (48, 216, 48), 2)
            cv2.putText(img, str(round(prediction[0][0]*100, 2)) + "%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (48, 216, 48), 2)
        elif np.argmax(prediction) == 1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, "No mask", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, str(round(prediction[0][1]*100, 2)) + "%", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(np.argmax(prediction))
        print(tf.gather(labels, np.argmax(prediction)))
    
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xff == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()