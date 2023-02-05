import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

def make_generator_model():
    model = tf.keras.Sequential()
    #model.add(layers.Dense(256*3*256, use_bias=False, input_shape=(3,256,256)))

    print("-1")
    model.add(layers.Conv2DTranspose(10, (10, 10), input_shape=(256,256,3), strides=(1, 1), padding='same', use_bias=False))
    
    # assert model.output_shape == (None, 10, 256, 256) #10 256 256
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print("0")
    # model.add(layers.Reshape((7, 7, 256)))
    # assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(20, (30, 30), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 128, 256, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print("1")

    model.add(layers.Conv2DTranspose(30, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 64, 512, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print("2")
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    print("3")
    #assert model.output_shape == (None, 128, 1024, 1)
    #print(model.output_shape)
    return model