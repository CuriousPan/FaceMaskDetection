# Face Mask Detection Project

This is a convolutional neural network which identifies whether a person has a face mask or notin the real time using webcamera.

## Used technologies

* Python 3.7.x 
* Tensorfow and Keras 2.3.1
* OpenCV
* Pickle
* Numpy
* sklearn (depricated)

## Setup

* Install any version of Python 3.7
* Install all required libraries

```
pip install tensorflow
pip install numpy
pip install opencv-python
pip install sklearn
```

***Note 1:*** keras is included in Tensorflow, however there can be some problems. In this case install keras separetly.

***Note 2:*** make sure all versions of listed libraries are compatible with each other.

## Getting started

***LoadingData.py***

Imports
```python
import numpy as np
import os
import cv2
import pickle
from sklearn.utils import shuffle
import keras
```

Firstly, download image dataset. I used [image dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) from [Kaggle.com](https://www.kaggle.com/).

Then prepare it in readable for CNN format. Here we use keras, OpenCV, Numpy and sklearn libraries. 

```python
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
```
 
After that save the data in the pickle format. It will be used later when we will unpack the data and fit it in the CNN.
```python
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
```

***ConvNet.py***

In this file CNN is defined.

Imports

```python
import tensorflow as tf
import keras
import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
```

Load the dataset in a normal format.

```python
#path to your data in pickle format
os.chdir("C:\PythonStuff\MasksProject\savedData")
train_x = pickle.load(open("train_x.pickle", "rb"))
train_y = pickle.load(open("train_y.pickle", "rb"))
```

Then the neural is defined.

```python
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
```

Compile and train with our data.

```python
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_split=0.1)
```

Here is the result of training. 
***Note:*** It make take you a several tries to "catch" the decent results for validation accuracy and validation loss.

![accuracy_image](C:\Users\CuriousPan\Desktop\accuracy.png)


