import tensorflow as tf
import keras
import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

#path to your data in pickle format
os.chdir("C:\PythonStuff\MasksProject\savedData")
train_x = pickle.load(open("train_x.pickle", "rb"))
train_y = pickle.load(open("train_y.pickle", "rb"))

model = Sequential([
 
	Conv2D(filters=16, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same", input_shape=train_x.shape[1:]), 
    MaxPooling2D(pool_size=(2,2), padding="same"),   
    
    Conv2D(filters=32, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same"),
    MaxPooling2D(pool_size=(2,2), padding="same"),
    
    Conv2D(filters=64, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same"),
    MaxPooling2D(pool_size=(2,2), padding="same"),
    
    Conv2D(filters=128, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same", input_shape=train_x.shape[1:]), 
    MaxPooling2D(pool_size=(2,2), padding="same"),   
    
    Conv2D(filters=256, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same"),
    MaxPooling2D(pool_size=(2,2), padding="same"),
    
    Conv2D(filters=512, kernel_size=(3,3), activation="relu", strides=(1,1), padding="same"),
    MaxPooling2D(pool_size=(2,2), padding="same"),
    
    Flatten(),
    Dropout(0.6),
    
    Dense(units=512, activation="relu"),
    Dropout(0.5),
    
    Dense(units=2, activation="softmax"),  
  
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.1)

os.chdir("C:\PythonStuff\MasksProject\model")
model.save("model.h5")