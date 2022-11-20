
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
from opts import parse_opts 
from data_generator_TL import DataGenerator
from utils_acc import AverageMeter, calculate_accuracy, calculate_precision_and_recall
import pickle
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)

df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")

df = df.drop([499])
df_train = df.iloc[:597, :]
df_val =df.iloc[597:, :]

# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
mask_list_train = df_train["Mask"].to_list()
mask_list_val = df_val["Mask"].to_list()


params = {'dim': (96, 192, 192),
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}


def train(training_generator, validation_generator, model, optimizer, total_epochs, criterion):
     # initialize tracker for minimum validation loss
    #early_stopper = EarlyStopper(patience=3, min_delta=10)
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    for epoch in range(total_epochs):
        # initialize the variables to monitor training and validation loss
        trainloss = 0.0
        trainacc = 0.0
        valloss = 0.0
        valacc = 0.0
        start_time = time.time()
        # training the model #
        for batch_idx, (data, target) in enumerate(training_generator):
            # move to GPU
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            trainloss+= loss.item()
            trainacc += calculate_accuracy(output, target)
        train_loss = trainloss/len(training_generator)
        train_acc = trainacc/len(training_generator)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print('burada')
        # validating the model #
        #model.eval()
        for _, (data, target) in enumerate(validation_generator):
            data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
            output = model(data)
            with torch.autocast('cuda'):
                loss = criterion(output, target)
            valloss += loss.item()
            valacc += calculate_accuracy(output, target)
        val_loss = valloss/len(validation_generator)
        val_acc = valacc/len(validation_generator)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Time: {:.3f} ...Epoch: {}.......Train_loss: {:.3f}.......Train_acc: {:.3f}.......Val_loss: {:.3f}.......Val_acc:{:.3f}'.format(time_elapsed, epoch, train_loss, train_acc, val_loss, val_acc))
    history = {"acc":train_acc_list, "loss":train_loss_list, "val_acc":val_acc_list, "val_loss":val_loss_list }
    with open('TL_Resnet2.pkl', 'wb') as f:
            pickle.dump(history, f)
    return model     
    
def main():
      
    training_generator = DataGenerator(list(dict_train.keys()), mask_list_train, dict_train, augment = False, batch_size=8, **params)
    validation_generator = DataGenerator(list(dict_val.keys()), mask_list_val, dict_val, augment = False, batch_size=8, **params)
    # getting model
    opt = parse_opts()   
    #device = torch.device("cuda") 
    criterion = nn.CrossEntropyLoss()
    # getting model
    #torch.manual_seed(sets.manual_seed)
    model = generate_model(opt)
    model = load_pretrained_model(model, opt.pretrain_path, 3)
    dev="cuda"
    model.to(dev)
    #model = make_data_parallel(model, opt.distributed, opt.device)
    parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)

    #optimizer = torch.optim.SGD(params= model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.9)

           # training
    train(training_generator, validation_generator, model, optimizer, 100, criterion) 
    # training

main()