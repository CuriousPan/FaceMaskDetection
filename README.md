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

***Note 1:*** keras is included in tensorflow, however there can be some problems. In this case install keras separetly.

***Note 2:*** make sure all versions of listed libraries are compatible with each other.

## Getting started

Firstly, download image dataset. I used [image dataset](https://www.kaggle.com/omkargurav/face-mask-dataset) from [Kaggle.com](https://www.kaggle.com/).

Then prepare it in readable for CNN format. Here we use OpenCV, Numpy and sklearn libraries. 

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
 



