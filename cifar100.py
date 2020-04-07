import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batchsize = 20
transform_train = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net = Net()
# net.to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print(optimizer.param_groups[0]['lr'])







# for epoch in range(3):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader,1):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         if i%2000 == 0:
#             print('[%d, %5d] loss: %.3f' % (epoch+1, i, running_loss/2000))
#             running_loss = 0.
# print('finished training')

# class_correct = [0 for i in range(10)]
# class_total = [0 for i in range(10)]

# with torch.no_grad():
#     for data in testloader:
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted==labels).squeeze()
#         for i in range(batchsize):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
            
# for i in range(10):
#     print('Accuracy of %5s: %.2f%%' % (classes[i], 100*class_correct[i]/class_total[i]))
    

        

