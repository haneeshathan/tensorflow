# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:23:32 2019

@author: htha930
"""

import tensorflow as tf
import keras as ke

import numpy as np

import matplotlib.pyplot as plt


fashion_mnist = ke.datasets.fashion_mnist

(X_train,y_train) , (X_test,y_test) = fashion_mnist.load_data()


#plottig one image from the dataset

plt.figure()

plt.imshow(X_train[10])

plt.colorbar()

plt.grid(False)

plt.show()


class_labels =['Shirt','Trouser','pullover','dress','coat','sandal','shirt','Sneaker','Bag','Ankle Boot']

plt.figure(figsize=(5,5))




for fig in range(25):
    plt.subplot(5,5,fig+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[fig])
    plt.xlabel(class_labels[y_train[fig]])
    plt.grid(False)
plt.show()

X_train = X_train/255.0


#

#define model
model_keras = ke.Sequential([ke.layers.Flatten(input_shape=(28,28)),ke.layers.Dense(128,activation=tf.nn.relu),ke.layers.Dense(10,activation=tf.nn.softmax)])


#compile the model
model_keras.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


model_keras.fit(X_train,y_train,epochs=5)

test_acc,test_loss = model_keras.evaluate(X_test,y_test)