import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms,utils
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
# plt.ion()   # 交互式模式

# landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
# # print(landmarks_frame)
# n = 65
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n, 1:].values
# landmarks = landmarks.astype('float').reshape(-1, 2)
# # print('Landmarks shape: {}'.format(landmarks.shape))
# plt.figure()
# image = io.imread(os.path.join('data/faces', img_name))
# plt.imshow(image)
# plt.scatter(landmarks[:,0], landmarks[:,1], s=50, marker='.', c='r')
# plt.savefig('a.png')

class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self,idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        sample = {'image':image, 'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces')

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size,(int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # print('rescale:',new_h,new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks*[new_w/w, new_h/h]
        return {'image':img, 'landmarks':landmarks}
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size,(int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # print('crop:',new_h,new_w)
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image = image[top:top+new_h, left:left+new_w]
        landmarks = landmarks - [left, top]
        return {'image':image, 'landmarks':landmarks}
class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image), 'landmarks':torch.from_numpy(landmarks)}

tra_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces', transform = transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
# for i in range(3):
#     sample = tra_dataset[i]
#     print(i, sample['image'].size(), sample['landmarks'].size())
dataloader = DataLoader(tra_dataset, batch_size=4, shuffle=True, num_workers=4)
def show_landmarks_batch(sample_batched):
    img_batch, lan_batck = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(img_batch)
    im_size = img_batch.size(2)
    grid = utils.make_grid(img_batch)
    plt.imshow(grid.numpy().transpose((1,2,0)))
    for i in range(batch_size):
        plt.scatter(lan_batck[i,:,0].numpy() + i*im_size, lan_batck[i,:,1].numpy(), s=10, marker='.', c='r')
        plt.title('batch from dataloader')
for i, sample_batched in enumerate(dataloader):
    print(i, sample_batched['image'].size(), sample_batched['landmarks'].size())
    plt.figure()
    show_landmarks_batch(sample_batched)
    plt.savefig('a.png')
    break

# scale = Rescale(256)
# crop = RandomCrop(128)
# composed = transforms.Compose([Rescale(256), RandomCrop(224)])
# fig = plt.figure()
# sample = face_dataset[65]
# print(sample['image'].shape)

# tra = composed(sample)
# print(tra['image'].shape)
# plt.imshow(tra['image'])
# plt.scatter(tra['landmarks'][:, 0], tra['landmarks'][:, 1], s=10, marker='.', c='r')
# plt.savefig('a.png')

# fig = plt.figure()
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#     print(i, type(sample['image']), sample['landmarks'].shape)
#     ax = plt.subplot(1,4,i+1)
#     # plt.tight_layout()
#     ax.set_title('sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['image'])
#     plt.scatter(sample['landmarks'][:, 0], sample['landmarks'][:, 1], s=10, marker='.', c='r')
#     if i==3:
#         plt.savefig('a.png')
#         break
    
        