import torch,os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import sys
sys.path.append("./d2l_pytorch") 
import d2l

import argparse
import logging
import yaml
import time



class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output
model = Model(2, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
for param_group in optimizer.param_groups:
    print(param_group['lr'])


# parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
# parser.add_argument('--work-path', default='data', type=str)
# parser.add_argument('--resume', action='store_true',
#                     help='resume from checkpoint')

# args = parser.parse_args()
# if args.work_path:
#     print(args.work_path)


# input_size = 5
# output_size = 2

# batch_size = 30
# data_size = 100 

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class RandomDataset(Dataset):
#     def __init__(self, size, length):
#         self.len = length
#         self.data = torch.randn(length,size)
        
#     def __getitem__(self, index):
#         return self.data[index]
    
#     def __len__(self):
#         return self.len
# rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True, num_workers=2)

# class Model(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)

#     def forward(self, input):
#         output = self.fc(input)
#         print("\tIn Model: input size", input.size(),
#               "output size", output.size())
#         return output
# model = Model(input_size, output_size)
# if torch.cuda.device_count():
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model.to(device)

# for data in rand_loader:
#     input = data.to(device)
#     output = model(input)
#     print("Outside: input size", input.size(),
#           "output_size", output.size())