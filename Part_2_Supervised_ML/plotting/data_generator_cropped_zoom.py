import os
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
import SimpleITK as stk
import random
from scipy import ndimage
from skimage.transform import resize
import torch

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, list_Masks, labels, batch_size, dim, n_channels, n_classes, shuffle, augment):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_Masks = list_Masks
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_Masks_temp = [self.list_Masks[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_Masks_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def crop_with_mask(self, mask_dat, ct_dat):

        for i in range(mask_dat.ndim):
            ct_dat = np.swapaxes(ct_dat, 0, i)  # send ct i-th axis to front
            mask_dat= np.swapaxes(mask_dat, 0, i)  # send mask i-th axis to front
            while np.all(mask_dat[0] == 0):
                ct_dat = ct_dat[1:]    # Crop CT where all mask values are zero in that axis
                mask_dat = mask_dat[1:]

            while np.all(mask_dat[-1] == 0):
                ct_dat = ct_dat[:-1]  # Crop CT where all mask values are zero in that axis
                mask_dat = mask_dat[:-1]

            ct_dat = np.swapaxes(ct_dat, 0, i)
            mask_dat = np.swapaxes(mask_dat, 0, i)

        return ct_dat


    def __data_generation(self, list_IDs_temp, list_Masks_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (ct_path, mask_path) in enumerate(zip(list_IDs_temp, list_Masks_temp)):
            # Store sample
            ct = stk.ReadImage(ct_path)
            ct= stk.GetArrayFromImage(ct)
            mask = stk.ReadImage(mask_path)
            mask= stk.GetArrayFromImage(mask)
            ct = self.crop_with_mask(mask, ct)
            ct = np.flip(ct)
            rot = random.choice(range(-20, 20))
            if self.augment:
              if i % 2:
                ct = ndimage.rotate(ct, rot, axes=(1,2))
              if i % 3:
                ct = np.flip(ct, axis=(0, 2))

            max_dim = np.max((ct.shape[1], ct.shape[2]))

            zoom_z = [self.dim[0] if ct.shape[0] > self.dim[0] else ct.shape[0]][0]
            zoom_y = [int(np.round(ct.shape[1]*self.dim[1]/max_dim)) if max_dim > self.dim[1] else ct.shape[1]][0]
            zoom_x = [int(np.round(ct.shape[2]*self.dim[2]/max_dim)) if max_dim > self.dim[2] else ct.shape[2]][0]
            ct = resize(ct, (zoom_z, zoom_y, zoom_x), preserve_range=True)
            
            z_pad = self.dim[0] - ct.shape[0]
            y_pad = self.dim[1] - ct.shape[1]
            x_pad = self.dim[2] - ct.shape[2]
        
            ct = np.pad(ct, ((int(np.floor(z_pad / 2)), int(np.ceil(z_pad / 2))),
                             (int(np.floor(y_pad / 2)), int(np.ceil(y_pad / 2))),
                             (int(np.floor(x_pad / 2)), int(np.ceil(x_pad / 2)))),
                        'constant', constant_values=0)
                        
            ct_min = -1024
            ct_max = 400        
            ct = ct.astype(np.float32)
            ct[ct > ct_max] = ct_max
            ct[ct < ct_min] = ct_min
            ct += -np.min(ct)
            ct /= np.max(ct)
            #plt.imshow(np.squeeze(ct)[:, :, 60], cmap="gray")
            #plt.show()
            #X[i,] = np.expand_dims(ct, axis=-1)
            X[i,] = ct

            #Store class
            y[i] = self.labels[ct_path]
        #y = to_categorical(y, num_classes=self.n_classes)
        y = torch.LongTensor(y)
   
        return X, y