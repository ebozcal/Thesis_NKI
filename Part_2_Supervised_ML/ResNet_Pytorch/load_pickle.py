import pickle
import pandas as pd

from data_generator import DataGenerator

params = {'dim': (96, 192, 192),
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")

df = df.drop([499])
df_train = df.iloc[:20, :]
df_val =df.iloc[20:35, :]

# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
mask_list_train = df_train["Mask"].to_list()
mask_list_val = df_val["Mask"].to_list()

training_generator = DataGenerator(list(dict_train.keys()), mask_list_train, dict_train, augment = False, batch_size=2, **params)
validation_generator = DataGenerator(list(dict_val.keys()), mask_list_val, dict_val, augment = False, batch_size=2, **params)
for key, items in training_generator:
    print("len_training_:", items)

#with open('/home/ertugrul/TL_Resnet.pkl', 'rb') as f:
#    history = pickle.load(f)
#print(history)
