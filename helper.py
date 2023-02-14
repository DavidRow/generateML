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
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

#seed with value of 10 for reproducibility
seed = 10
np.random.seed(seed)

#loads jpgs/pngs into np array for saving
def loadData(directory, name):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = Image.open(os.path.join(directory, filename))
            image_array = np.array(image)
            image_list.append(image_array)

    image_np = np.stack(image_list)
    np.save(name, image_np)

    
    

#split the data into traning and testing
def splitData(patingsArr,photosArr):
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
    return (test_data_photos, training_data_photos, test_data_patings, training_data_patings)


#loads npy files into arrays
def loadArray():
    photos = np.load('photo_jpg/photos.npy')
    patings = np.load('patings/patings.npy')
    return photos,patings


def printImages(photo, generated):
    _ , axarr= plt.subplots(1,2)
    axarr[0].imshow(photo)
    axarr[1].imshow(generated)
    plt.show()