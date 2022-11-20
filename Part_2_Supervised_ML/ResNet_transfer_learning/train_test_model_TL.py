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
#from data_generator_masked import DataGenerator
from data_generator_TL import DataGenerator
from early_stop import EarlyStopper
from utils_acc import AverageMeter, calculate_accuracy, calculate_precision_and_recall
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sn
from model_FE import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from opts import parse_opts 

#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")
df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_762_paths_labels_input_10_p1.csv")

#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_10.csv")

#df = df.drop([499])

df = df.drop([477])
df_train = df.iloc[:502, :]
df_val =df.iloc[502:633, :]
df_test =df.iloc[633:, :]


# Datasets
dict_train = pd.Series(df_train["label"].values, index=df_train["Image"].values).to_dict()
dict_val = pd.Series(df_val["label"].values, index=df_val["Image"].values).to_dict()
dict_test = pd.Series(df_test["label"].values, index=df_test["Image"].values).to_dict()
mask_list_train = df_train["Mask"].to_list()
mask_list_val = df_val["Mask"].to_list()
mask_list_test = df_test["Mask"].to_list()


params = {'dim': (96, 192, 192), # size for cropped CT
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

training_generator = DataGenerator(list(dict_train.keys()), mask_list_train, dict_train, augment = False, batch_size=8, **params)
validation_generator = DataGenerator(list(dict_val.keys()), mask_list_val, dict_val, augment = False, batch_size=8, **params)
test_generator = DataGenerator(list(dict_test.keys()), mask_list_test, dict_test, augment = False, batch_size=8, **params)

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
optimizer = torch.optim.SGD(params= model.parameters(), lr=0.001, momentum=0.9)


     # initialize tracker for minimum validation loss
early_stopper = EarlyStopper(patience=50, min_delta=0.01)
acc_list =[]
loss_list = []
val_acc_list = []
val_loss_list = []
for epoch in range(200):
    start_time = time.time()
    # initialize the variables to monitor training and validation loss
    trainloss = 0.0
    trainacc = 0.0
    valloss = 0.0
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
        trainloss+= loss.item()
        trainacc += calculate_accuracy(output, target)
    train_loss = trainloss/len(training_generator)
    train_acc = trainacc/len(training_generator)
    acc_list.append(train_acc)
    loss_list.append(train_loss)
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
    end_time = time.time()
    time_elapsed = end_time - start_time
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    print("TL_50_crp_762_SGD_Epoch:{},... Time:{}.....Train_loss:{}....Train_acc:{}....val_loss:{}.....val_acc:{}".format(epoch, round(time_elapsed,2),  round(train_loss, 4), round(train_acc, 4), round(val_loss, 4),  round(val_acc, 4)))

history_dict = {"acc":acc_list, "loss":loss_list, "val_acc":val_acc_list, "val_loss":val_loss_list}
with open('REsNet50_TL_762_SGD_crp_history_FT.pkl', 'wb') as f:
    pickle.dump(history_dict, f)
# Testing the model's performance
y_pred = []
y_true = []  
model.eval()
for data, target in test_generator:
    data, target = torch.Tensor(data).cuda(), torch.Tensor(target).cuda()
    output = model(data) # Feed Network
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
        
    labels = target.data.cpu().numpy()
    y_true.extend(labels) # Save Truth
   
pred_label_dict = {"label":y_true, "pred":y_pred}
with open('REsNet50_762_TL_SGD_crp_label_pred.pkl', 'wb') as f:
    pickle.dump(pred_label_dict, f)
        
