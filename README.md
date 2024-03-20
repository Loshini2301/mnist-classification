# Convolutional Deep Neural Network for Digit Classification
### NAME:LOSHINI.G
### REFERENCE NUMBER:212223220051
### DEPARTMENT:IT
## AIM
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model
<img width="708" alt="Screenshot 2024-03-20 143913" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/a530a779-0cb0-4357-b67d-65e1dfc06833">


## DESIGN STEPS


### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict
## PROGRAM

### Name:LOSHINI.G
### Register Number:212223220051
### LIBRARY IMPORTING:
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
```
### SHAPING:
```
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
```
### ONE HOT ENCODING:
```
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
```
### CNN MODEL:
```
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=40,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(layers.Conv2D(filters=80,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(12,activation='relu'))
model.add(layers.Dense(14,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train_scaled, y_train_onehot, epochs=5, batch_size=64,
          validation_data=(X_test_scaled, y_test_onehot))
```
### METRICES:
```
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy', 'val_accuracy']].plot()
metrics[['loss', 'val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test, x_test_predictions))
print(classification_report(y_test, x_test_predictions))
```
### PREDICTION:
```
img = image.load_img('/content/Screenshot 2024-03-20 135143.png')
type(img)
img = image.load_img('/content/Screenshot 2024-03-20 135143.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor, (28, 28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy() / 255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1, 28, 28, 1)),
    axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28, 28), cmap='gray')
img_28_gray_inverted = 255.0 - img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy() / 255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1, 28, 28, 1)),
    axis=1)
print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="316" alt="Screenshot 2024-03-20 160714" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/3eaf646f-0c31-4360-bac3-67dc30b08985">


![image](https://github.com/Loshini2301/mnist-classification/assets/150007305/6b4bcf7a-2362-4145-94b6-5178d95df4c2)

![image](https://github.com/Loshini2301/mnist-classification/assets/150007305/dfa58696-1255-499d-a46b-5e25f2206b59)



### Classification Report

<img width="323" alt="Screenshot 2024-03-20 154058" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/a302615a-bede-4e97-8650-9f6a5ecd28bf">


### Confusion Matrix

<img width="322" alt="Screenshot 2024-03-20 154156" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/bb3ca622-dca1-4320-8e30-da1f95f695d1">


### New Sample Data Prediction
<img width="314" alt="Screenshot 2024-03-20 155728" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/86ead004-1395-4abd-9476-2ab251c5f854">
<img width="322" alt="image" src="https://github.com/Loshini2301/mnist-classification/assets/150007305/e117da0f-1384-4c9e-975e-2fd4d4e54178">


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
