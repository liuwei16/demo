import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=False,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=False,
    train=False,
    transform=transform)

# dataloaders
batch_size = 16
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# def imshow(img,labels):
#     img = img/2+0.5
#     plt.imshow(img.numpy().transpose((1,2,0)))
#     plt.title(' '.join('%5s' % classes[labels[j]] for j in range(16)))
#     plt.savefig('a.png')
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# print(images.size())
# imshow(torchvision.utils.make_grid(images), labels)
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x

dummy_input = torch.autograd.Variable(torch.rand(batch_size, 1, 28, 28))
model3 = Net3()
print(model3)
with SummaryWriter(comment='log/_fashionmnist_net3') as w:
    w.add_graph(model3, (dummy_input, ))
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model3.to(device)  #or = model2
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

writer = SummaryWriter(comment='log/_fashionmnist_logs')
num_epochs = 10
num_batches = len(trainloader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for step, data in enumerate(trainloader):
        n_iter = epoch * num_batches + step
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        out = net(images)
        loss_value = loss(out, labels)
        loss_value.backward()
        optimizer.step()
        writer.add_scalar('loss', loss_value.item(), n_iter)
        running_loss += loss_value.item()
        if step % 500 == 499:    # 每 500 个 mini-batches 就输出一次训练信息
            print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 500))
            running_loss = 0.
writer.close()
print('Finished Training')
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        


# writer = SummaryWriter('log/1')
# writer.add_image('four_fashion_img', torchvision.utils.make_grid(images))
# # writer.add_graph(net, images)
# # writer.close()

# def select_n_random(data, labels, n= 100):
#     assert len(data)==len(labels)
#     perm = torch.randperm(len(data))
#     return data[perm][:n], labels[perm][:n]
# images, labels = select_n_random(trainset.data, trainset.targets)
# class_labels = [classes[lab] for lab in labels]
# features = images.view(-1,28*28)
# writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
# writer.close()