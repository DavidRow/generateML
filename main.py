import os
import helper
import tensorflow as tf
import GAN as gan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from   pytorch_lightning import LightningDataModule, LightningModule, Trainer
from   pytorch_lightning.callbacks.progress import TQDMProgressBar
from   torch.utils.data import DataLoader, random_split
from   torchvision.datasets import MNIST
import tenserGAN 
from PIL import Image
import matplotlib.pyplot as plt

#seed with value of 10 for reproducibility
seed = 10
np.random.seed(seed)


#load the arrays for use
photo_jpg, patings_jpg = helper.loadArray()

#split them into training and test
test_data_photos, training_data_photos, test_data_patings, training_data_patings = helper.splitData(patings_jpg, photo_jpg)


# make GAN
generator = tenserGAN.make_generator_model()    
generated_image = generator(training_data_photos[0:1], training=False)

# make racist 
discriminator = tenserGAN.make_discriminator_model()

# optimizers?
generator_optimizer = tf.keras.optimizers.Adam(1e-4)    
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Training time baby
epochs = 1
newgenerator = tenserGAN.train(generator, discriminator, generator_optimizer, discriminator_optimizer, epochs, training_data_photos, training_data_patings)

# generate an image as a test
generatedpainting = newgenerator(training_data_photos[0:1], training=False)
tenserGAN.printImages(training_data_photos[0], generatedpainting )











