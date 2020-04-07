# import torch,os
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader,Dataset
# import sys
# import argparse
# import logging
# import yaml
# import time
# from easydict import EasyDict
# from models import *
# from utils import Logger, count_parameters, data_augmentation, \
#     load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
#     save_checkpoint, adjust_learning_rate, get_current_lr
import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.f1 = nn.Linear(4, 1, bias=True)
        self.f2 = nn.Linear(1, 1, bias=True)
        self.weight_init()
    def forward(self, input):
        self.input = input
        output = self.f1(input)       
        output = self.f2(output)      
        return output
    def weight_init(self):
        self.f1.weight.data.fill_(8.0)    
        self.f1.bias.data.fill_(2.0)   
    '定义钩子函数，打印出需要的量，注意钩子函数的形参是固定的'
    def my_pre_hook(self,module,input):
        print('------pre_hook-----------')
        print('module: ',module)
        print('input: ',input)
    def my_forward_hook(self,module,input,output):
        print('\n------forward_hook-----------')
        print('input: ',input)
        print('output: ',output)
    def my_backward_hook(self, module, grad_input, grad_output):
        print('\n------backward_hook-----------')
        print('grad_input: ', grad_input)
        print('grad_output: ', grad_output)
        return grad_input
   
if __name__ == '__main__':
    input = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
    net = MyNet()
    'hook函数要net(input)执行前执行，因为hook函数是在forward/backward的时候进行绑定的'
    '注册钩子的时候，传入钩子函数的函数名即可'
    net.register_forward_pre_hook(net.my_pre_hook)   
    net.register_forward_hook(net.my_forward_hook)
    net.register_backward_hook(net.my_backward_hook)
    result = net(input)
    result.backward()


# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
# logger = Logger(log_file_name='{}.txt'.format(timestamp),
#                 log_level=logging.DEBUG, logger_name="CIFAR").get_log()

# logger.info('Training')
# logger1 = logging.getLogger("CIFAR.inference")
# logger1.info('test')

# parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
# parser.add_argument('--work-path',default='experiments/cifar10/lenet/', type=str)
# parser.add_argument('--resume', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

# with open(args.work_path + '/config.yaml') as f:
#     config = yaml.load(f)

# config = EasyDict(config)
# net = get_model(config)
# device = 'cuda' if config.use_gpu else 'cpu'
# net.to(device)
# # define loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     net.parameters(),
#     config.lr_scheduler.base_lr,
#     momentum=config.optimize.momentum,
#     weight_decay=config.optimize.weight_decay,
#     nesterov=config.optimize.nesterov)
# # resume from a checkpoint
# last_epoch = -1
# best_prec = 0
# for epoch in range(last_epoch + 1, config.epochs):
#     lr = adjust_learning_rate(optimizer, epoch, config)
#     for param_group in optimizer.param_groups:
#         print(param_group['lr']) 
            
            
