import cv2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import numpy as np
from numpy import load

data = load('Vehicles_training_features_newer.npz')
training_features =data['arr_0']
data = load('Vehicles_training_labels_newer.npz')
training_labels =data['arr_0']
training_labels = tf.keras.utils.to_categorical(training_labels, 8)


def model_builder():
    model = Sequential()
    model.add(Conv2D(64,(3,3),activation='relu',input_shape=training_features.shape[1:]))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(8,activation='softmax'))
    model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model

model = model_builder()
model.fit(training_features,training_labels,epochs=10,batch_size=8,validation_data=0.1)
model.save('VehicleClassifierCustomBuilt.h5')