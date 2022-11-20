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
        y = torch.zeros((self.batch_size), dtype=torch.float)

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

            assert isinstance(ct, (np.ndarray, np.generic))
            assert (ct.ndim - 1 == len(self.dim) or ct.ndim == len(self.dim)), \
            rank = len(self.dim)
            # Create placeholders for the new shape
            from_indices = [[0, image.shape[d]] for d in range(rank)]
            to_padding = [[0, 0] for d in range(rank)]
            slicer = [slice(None)] * rank
    # For each dimensions find whether it is supposed to be cropped or padded
            for i in range(rank):
                if image.shape[i] < img_size[i]:
                    to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
                    to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
                else:
                    from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
                    from_indices[i][1] = from_indices[i][0] + img_size[i]
        # Create slicer object to crop or leave each dimension
                slicer[i] = slice(from_indices[i][0], from_indices[i][1])
    # Pad the cropped image to extend the missing dimension
            ct = np.pad(image[slicer], to_padding)
            
            #fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
            #axes = [ax1, ax2,ax3,ax4,ax5, ax6]

            #for i in range(6):
             #   axes[i].imshow(np.squeeze(ct)[:, :, 50+i], cmap="gray")
            #plt.show()

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
            y[i] =self.labels[ct_path]
      
        #y = to_categorical(y, num_classes=self.n_classes)
        #y = torch.LongTensor(y)
       
        return X, y

        