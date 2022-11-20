# Build model.
import numpy as np
#import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
from define_model import get_model
from data_generator2 import DataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from mplot import plot_result


df_train = pd.read_csv("/processing/ertugrul/Part_2/df_426__withlabel_mask.csv").iloc[:327, :]
df_val = pd.read_csv("/processing/ertugrul/Part_2/df_426__withlabel_mask.csv").iloc[327:, :]
#df_test = pd.read_csv("/processing/ertugrul/Part_2/Labeling/CNN_823_paths_labels.csv").iloc[150:200, :]
#lab_train = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
#lab_val = [0, 1, 2, 0, 1]
# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
#dict_test = pd.Series(df_test["label"].values, index=df_test["Image"].values).to_dict()
#dict_train = pd.Series(lab_train, index=df_train["Image"].values).to_dict()
#dict_val = pd.Series(lab_val, index=df_val["Image"].values).to_dict()

params = {'dim': (192, 192, 96),
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

training_generator = DataGenerator(list(dict_train.keys()),  dict_train, augment = False, batch_size=8, **params)
validation_generator = DataGenerator(list(dict_val.keys()), dict_val, augment = False, batch_size=8, **params)
    
model = get_model(192, 192, 96)
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
metrics=["acc"])
checkpoint_cb = keras.callbacks.ModelCheckpoint("3d_image_classification.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

epochs= 10
#Train model on dataset
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              workers=10,
                              epochs=epochs,
                              verbose=1)
plot_result(history)
                              


