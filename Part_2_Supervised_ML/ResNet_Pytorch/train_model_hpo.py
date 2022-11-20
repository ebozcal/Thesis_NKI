import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
import numpy as np
from scipy import ndimage
import os
import pandas as pd
from utils.logger import log
from data_generator_cropped import DataGenerator
from early_stop import EarlyStopper
from utils_acc import AverageMeter, calculate_accuracy, calculate_precision_and_recall
import pickle
from define_model_cropped import generate_ResNet
import wandb

df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")

df = df.drop([499])
df_train = df.iloc[:97, :]
df_val =df.iloc[97:150, :]

# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
mask_list_train = df_train["Mask"].to_list()
mask_list_val = df_val["Mask"].to_list()


params = {'dim': (96, 192, 192),
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        'opt': {'values': ['Adam', 'SGD']},
        'lr': {'max': 0.01, 'min': 0.0001},
        'bs': {'values': [8, 16]},
        'model_depth':{'values' :[50]}
        } 
     }

sweep_id = wandb.sweep(sweep_configuration, project='HPO_Resnet_cropped')

def train(training_generator, validation_generator, model, optimizer, total_epochs, criterion):
     # initialize tracker for minimum validation loss
    early_stopper = EarlyStopper(patience=50, min_delta=0.01)
    for epoch in range(total_epochs):
        # initialize the variables to monitor training and validation loss
        trainloss = 0.0
        valloss = 0.0
        trainacc = 0.0
        valacc = 0.0
        # training the model #
        for _, (data, target) in enumerate(training_generator):
            # move to GPU
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            wandb.log({"batch loss": loss.item()})
            trainloss+= loss.item()
            trainacc += calculate_accuracy(output, target)
        train_loss = trainloss/len(training_generator)
        train_acc = trainacc/len(training_generator)
        wandb.log({"loss": train_loss, "acc":train_acc})       
 
        # validating the model #
        model.eval()
        for _, (data, target) in enumerate(validation_generator):
            
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            output = model(data)
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            valloss += loss.item()
            valacc += calculate_accuracy(output, target)
        val_loss = valloss/len(validation_generator)
        if early_stopper.early_stop(val_loss):             
            break
        val_acc = valacc/len(validation_generator)
        print("Resnet_hpo_Epoch:{}....Train_loss:{}....Val_loss:{}....Train_acc:{}.....val_acc:{}".format(epoch,  round(train_loss, 4), round(val_loss, 4), round(train_acc, 4),  round(val_acc, 4)))
        wandb.log({"val_loss": val_loss, "val_acc":val_acc})
    
    return model     
    
def main(config=None):
    with wandb.init(config=config):
        config = wandb.config 
        training_generator = DataGenerator(list(dict_train.keys()), mask_list_train, dict_train, augment = True, batch_size=config.bs, **params)
        validation_generator = DataGenerator(list(dict_val.keys()), mask_list_val, dict_val, augment = False, batch_size=config.bs, **params)
        criterion = nn.CrossEntropyLoss()
    # getting model
        model = generate_ResNet(config.model_depth, config.bs)
        dev="cuda"
        model.to(dev)
    # optimizer
        if config.opt=="SGD":
            optimizer =  torch.optim.SGD(params= model.parameters(), lr=config.lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(params= model.parameters(), lr=config.lr)
    # training
        train(training_generator, validation_generator, model, optimizer, 200, criterion) 

wandb.agent(sweep_id, function=main, count=20)