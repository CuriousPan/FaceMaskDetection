import numpy as np
import os
import cv2
import pickle
from sklearn.utils import shuffle
import keras

labels = ["with_mask", "without_mask"]

train_x = []
train_y = []

#path to dataset
path = "C:\PythonStuff\MasksProject\data"  

IMG_SIZE = 100

for label in labels:
    data_path = os.path.join(path, label)
    y = labels.index(label)
    print(data_path)
   
    for img in os.listdir(data_path):
        try:
            img_array = cv2.imread(os.path.join(data_path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_x.append(new_array)
            train_y.append(y)
        except Exception as e:
            print("Error in reading image")

shuffle(train_x, train_y)

train_x = np.array(train_x, dtype="uint16").reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_y = keras.utils.to_categorical(train_y)

if not os.path.exists("savedData"):
    os.makedirs("savedData")

#path where data will be saved in pickle format
os.chdir("C:\PythonStuff\MasksProject\savedData")

pickle_out_x = open("train_x.pickle", "wb")
pickle.dump(train_x, pickle_out_x)    
pickle_out_x.close()

pickle_out_y = open("train_y.pickle", "wb")
pickle.dump(train_y, pickle_out_y)
pickle_out_y.close()  