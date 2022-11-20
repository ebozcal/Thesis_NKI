#import pickle

#with open('/home/ertugrul/TL_Resnet.pkl', 'rb') as f:
 #   history = pickle.load(f)
#print(history)
from torchsummary import summary
import torch
from torch import nn
#from define_model import generate_ResNet
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)

from opts import parse_opts 

opt = parse_opts()
  
model = generate_model(opt)
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
c=0
for k, v in model.state_dict().items():
    print(k, v)
    c+=1
    if c >2:
        break
model = load_pretrained_model(model, opt.pretrain_path, 3)
#pretrain = torch.load("/processing/ertugrul/Part_2/TL_ResNet_pytorch_2/resnet_50.pth")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
c=0
for k, v in model.state_dict().items():
    print(k, v)
    c+=1
    if c >2:
        break
ert

#for k, v in pretrain['state_dict'].items():
 #   print(k, v.shape)
#print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#for k, v in model.state_dict().items():
 #   print(k, v.shape)
net_dict = model.state_dict()
print(len(net_dict.items()))
print(len(pretrain['state_dict'].items()))

pretrain_dict = {k[7:]: v for k, v in pretrain['state_dict'].items() if k[7:] in net_dict}
#for k, l in pretrain_dict.items():
 #   print(k)
#print("pretrain", pretrain_dict)
model.load_state_dict(pretrain_dict)
model.fc = nn.Linear(512, 3)

#tmp_model = model
      
#tmp_model.fc = nn.Linear(tmp_model.fc.in_features, n_finetune_classes)

#parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)
#print(net_dict.keys())

print("tttttttttttttttttttttttttttttttt")
#for k, n in pretrain_dict.items():
#    print(k)
print(model.state_dict()["conv1.weight"].shape)
print("zzzzzzzzzzzzzzzzzzzzzzzzz")
print(model.state_dict()["fc.weight"].shape)

#print("pretrain_dict:", pretrain_dict.keys())


