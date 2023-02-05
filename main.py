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



#
#load the arrays for use
photo_jpg, patings_jpg = helper.loadArray()

#split them into training and test
test_data_photos, training_data_photos, test_data_patings, training_data_patings = helper.splitData(patings_jpg, photo_jpg)


# make GAN
#GAN = gan.RGAN(3, 256, 256)
#print(training_data_photos[0].shape)
#training_data_photos = training_data_photos
generator = tenserGAN.make_generator_model()    
generated_image = generator(training_data_photos[0:1], training=False)

plt.imshow(training_data_photos[0])
plt.show()
print(generated_image.shape)
plt.imshow(generated_image[0])
plt.show()
#GAN.training_step(torch.flatten(training_data_photos[1]) , 0)



