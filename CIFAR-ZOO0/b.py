import torch,os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import sys
import argparse
import logging
import yaml
import time
from easydict import EasyDict
from models import *
from utils import Logger, count_parameters, data_augmentation, \
    load_checkpoint, get_data_loader, mixup_data, mixup_criterion, \
    save_checkpoint, adjust_learning_rate, get_current_lr

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

global args, config, last_epoch, best_prec

parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
parser.add_argument('--work-path',default='myexe/', type=str)
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction

    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors

        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1

        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels, value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth

        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / (length - 1)

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))

        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')

# timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logger = Logger(log_file_name=args.work_path + '/logs_18_cos200_lsr.txt',
                log_level=logging.DEBUG, logger_name="CIFAR").get_log()
with open(args.work_path + '/config.yaml') as f:
    config = yaml.load(f)
# print(type(config))
config = EasyDict(config)
# print(type(config))
# logger.info(config)
net = get_model(config)
# print(count_parameters(net))
device = 'cuda' if config.use_gpu else 'cpu'
net.to(device)
# define loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = LSR()
optimizer = torch.optim.SGD(
    net.parameters(),
    config.lr_scheduler.base_lr,
    momentum=config.optimize.momentum,
    weight_decay=config.optimize.weight_decay,
    nesterov=config.optimize.nesterov)
# resume from a checkpoint
last_epoch = -1
best_prec = 0
if args.work_path:
    ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth'
    if args.resume:
        best_prec, last_epoch = load_checkpoint(
            ckpt_file_name, net, optimizer=optimizer)
# load training data, do data augmentation and get data loader
transform_train = transforms.Compose(
    data_augmentation(config))
transform_test = transforms.Compose(
    data_augmentation(config, is_train=False))
train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

def train(train_loader, net, criterion, optimizer, epoch, device):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total
    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc

logger.info("=======  Training  =======\n")
starttime = time.time()
for epoch in range(last_epoch + 1, config.epochs):
    lr = adjust_learning_rate(optimizer, epoch, config)
    train(train_loader, net, criterion, optimizer, epoch, device)
    if (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
        test(test_loader, net, criterion, optimizer, epoch, device)
costtime = time.time()-starttime
h = int(costtime/3600)
m = int((costtime%3600)/60)
logger.info(
        "======== Training Finished with {} h, {} m.   best_test_acc: {:.3f}% ========".format(h,m,best_prec))
            
            
