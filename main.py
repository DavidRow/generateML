import os

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
from PIL import Image

#seed with value of 10 for reproducibility
seed = 10
np.random.seed(seed)

#fetch the images and put them into an array
def loadData(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(directory, filename))
            image_array = np.array(image)
            image_list.append(image_array)

    return np.stack(image_list)
#patingsArr = loadData('patings')
#photosArr = loadData('photo_jpg')

#save it to the hard drive
#np.save('patings/patings.npy', patingsArr)
#np.save('photo_jpg/photo_jpg.npy', photosArr)

#load the arrays for use
patingsArr = np.load('patings/patings.npy')
photosArr  = np.load('photo_jpg/photo_jpg.npy')

#shuffle training and test
np.random.shuffle(patingsArr)
np.random.shuffle(photosArr)
test_proportion = 0.2

#get proprotions 
test_size_paint = int(test_proportion * len(patingsArr))
test_size_photos = int(test_proportion * len(photosArr))

#split testing data
test_data_patings = patingsArr[:test_size_paint]
training_data_patings = patingsArr[test_size_paint:]

#split photos data
test_data_photos = photosArr[:test_size_photos]
training_data_photos = photosArr[test_size_photos:]

#Generator
class generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape


#Discriminator(racist)
class racist(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
