import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import logging

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.cuda.get_device_name(0))

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

logger = Logger(log_file_name='log.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()


batch_size = 20
transform_train = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
# net.fc = nn.Sequential(
#     nn.Linear(num_ftrs, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
#     )

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(trainloader,1):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i%2000 == 0:
            logger.info('[%d, %5d] loss: %.3f' % (epoch+1, i, running_loss/2000))
            running_loss = 0.
logger.info('finished training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item() 
logger.info('Accuracy : %.2f%%' % (100*correct/total))

# class_correct = [0 for i in range(10)]
# class_total = [0 for i in range(10)]
# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted==labels).squeeze()
#         for i in range(batch_size):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
            
# for i in range(10):
#     print('Accuracy of %5s: %.2f%%' % (classes[i], 100*class_correct[i]/class_total[i]))
# logger.info('Accuracy : %.2f%%' % (100*sum(class_correct)/sum(class_total)))
        

