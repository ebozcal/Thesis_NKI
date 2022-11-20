import torch
from torch import nn
from torch import optim
import time
import numpy as np
import os
import pandas as pd
from utils.logger import log
from data_generator import DataGenerator
from early_stop import EarlyStopper
from define_model import  cnn3d3, cnn3d4
from test_regression import calculate_accuracy
import wandb

df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/COntinious_CNN_710_paths_labels_m_full_20.csv")
#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/Regression_CNN_791_paths_labels_m_full_10.csv")

#df = df.drop([499])
df_train = df.iloc[:553, :]
df_val =df.iloc[553:, :]

# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
mask_list_train = df_train["Mask"].to_list()
mask_list_val = df_val["Mask"].to_list()


params = {'dim': (96, 192, 192),
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': 
    {
        #'bn': {'values': ['bn', 'no_bn']},
        'lr': {'max': 0.05, 'min': 0.0005},
        'dr': {'max': 0.3, 'min': 0.1},
        'nl': {'values': [3, 4]},
        'nf': {'values': [8, 16]},
        'bs': {'values': [4, 8]},
        } 
     }

sweep_id = wandb.sweep(sweep_configuration, project='HPO_CNN_pytorch_regression_10perc')

def train(training_generator, validation_generator, model, optimizer, total_epochs, criterion):
     # initialize tracker for minimum validation loss
    early_stopper = EarlyStopper(patience=30, min_delta=0.01)
    for epoch in range(total_epochs):
        start_time = time.time()
        # initialize the variables to monitor training and validation loss
        trainloss = 0
        trainacc = 0
        valloss = 0
        valacc = 0
        # training the model #
        for _, (data, target) in enumerate(training_generator):
            # move to GPU
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            output = torch.reshape(output, (-1,))
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #wandb.log({"batch loss": loss.item()})
            #train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            trainloss+= loss.item()
            trainacc += calculate_accuracy(output, target)
        train_loss = trainloss/len(training_generator)
        train_acc = trainacc/len(training_generator)
        wandb.log({"train_loss": train_loss, "train_acc":train_acc})       
        # validating the model #
        #model.eval()
        for _, (data, target) in enumerate(validation_generator):
            
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            output = model(data)
            output = torch.reshape(output, (-1,))
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            valloss += loss.item()
            valacc += calculate_accuracy(output, target)
        val_loss = valloss/len(validation_generator)
        if early_stopper.early_stop(val_loss):             
            break
        val_acc = valacc/len(validation_generator)
        wandb.log({"val_loss": val_loss, "val_acc":val_acc})
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Epoch:{},... Time:{}.....Train_loss:{}......Train_acc:{}....Val_loss:{}......Val_acc:{}".format(epoch, round(time_elapsed,2),  round(train_loss, 3),  round(train_acc, 3), round(val_loss, 3), round(val_acc,3)))

    return model     
    
def main(config=None):
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config 
        training_generator = DataGenerator(list(dict_train.keys()), mask_list_train, dict_train, augment = False, batch_size=config.bs, **params)
        validation_generator = DataGenerator(list(dict_val.keys()), mask_list_val, dict_val, augment = False, batch_size=config.bs, **params)
        criterion = nn.MSELoss()
    # getting model
        if config.nl ==3:
            model =  cnn3d3(config.nf,  config.dr)
        if config.nl ==4:
            model =  cnn3d4(config.nf,  config.dr)
        dev="cuda"
        model.to(dev)
    # optimizer
       # if config.opt=="SGD":
        #    optimizer =  torch.optim.SGD(params= model.parameters(), lr=config.lr, momentum=0.9)
        #else:
        optimizer = torch.optim.Adam(params= model.parameters(), lr=config.lr)
    # training
        train(training_generator, validation_generator, model, optimizer, 200, criterion) 

wandb.agent(sweep_id, function=main, count=20)