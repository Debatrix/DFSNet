import os

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms


class BaseDataset(data.Dataset):
    def __init__(self,
                 path,
                 mode='train',
                 debug_data=False,
                 size=(640, 480),
                 seed=3141):
        np.random.seed(seed)

        self.path = path
        self.mode = mode

        if self.mode == 'train' or self.mode == 'val':
            with open(os.path.join(path, 'train.txt'), 'r') as f:
                img_list = [
                    tuple(line.strip().split(' ')) for line in f.readlines()
                ]
            np.random.shuffle(img_list)
            if self.mode == 'train':
                self.img_list = img_list[:int(
                    0.1 * len(img_list)
                )] if debug_data else img_list[:int(0.8 * len(img_list))]
            else:
                self.img_list = img_list[int(
                    0.95 * len(img_list)
                ):] if debug_data else img_list[int(0.8 * len(img_list)):]
        else:
            with open(os.path.join(path, 'test.txt'), 'r') as f:
                img_list = [
                    tuple(line.strip().split(' ')) for line in f.readlines()
                ]
                self.img_list = img_list[:int(0.1 * len(img_list)
                                              )] if debug_data else img_list
        self.all_imglist = img_list

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        self.Norm = transforms.Normalize(mean=[0.480],
                                         std=[0.200],
                                         inplace=False)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_name, score = self.img_list[item]
        score = torch.tensor(float(score), dtype=torch.float).view((-1))

        img = Image.open(os.path.join(self.path, 'Image', img_name))
        img = self.transform(img)
        return img, score


class monoSimDataset(BaseDataset):
    def __init__(self,
                 path,
                 mode='train',
                 debug_data=False,
                 size=(480, 640),
                 seed=3141,
                 upsample=False):
        super(monoSimDataset, self).__init__(path, mode, debug_data, size,
                                             seed)

        self.sim = sio.loadmat(os.path.join(path, 'sim.mat'))['sim']
        self.sim = (self.sim + 1) / 2

        with open(os.path.join(path, 'gallery.txt'), 'r') as f:
            self.gallery_dict = {
                x.strip().split(' ')[1]: x.strip().split(' ')[0]
                for x in f.readlines()
            }
        self.index = {x[0]: x[2] for x in self.all_imglist}

        self.transmask = transforms.Compose([
            transforms.Resize((size[0] // 4, size[1] // 4)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        img_name, img_label, x = self.img_list[item]
        y = self.index[self.gallery_dict[img_label]]
        score = self.sim[int(x), int(y)]
        score = torch.tensor(float(score), dtype=torch.float).view((-1))

        img = Image.open(os.path.join(self.path, 'Image', img_name))
        img = self.transform(img)
        img = self.Norm(img)
        if self.mode != 'test':
            mask = Image.open(
                os.path.join(self.path, 'Mask',
                             img_name.split('.')[0] + '.png'))
            mask = self.transmask(mask)
        else:
            mask = torch.ones_like(img)
        img = torch.cat((img, img, img))
        ret = (img, mask, score,
               img_name) if self.mode == 'test' else (img, mask, score)
        return ret


if __name__ == '__main__':
    data = monoSimDataset(path='data/cx2', debug_data=False)
    print(len(data))
    # print(data.label_list)
    for x, y, z in data:
        print(x.shape, y.shape, z)
