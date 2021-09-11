""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import random
import glob
import string
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import lmdb
import cv2


Records = namedtuple('Records', ['labels', 'insts'])

def pad_patch(min_x, min_y, max_x, max_y, output_size):
        width = max_x - min_x
        height = max_y - min_y

        if width < output_size:
            w_diff_left = (output_size - width) // 2
            w_diff_right = output_size - width - w_diff_left
            min_x = max(0, min_x - w_diff_left)
            max_x = max_x + w_diff_right
        if height < output_size:
            h_diff_top = (output_size - height) // 2
            h_diff_bot = output_size - height - h_diff_top
            min_y = max(0, min_y - h_diff_top)
            max_y = max_y + h_diff_bot

        return min_x, min_y, max_x, max_y


def write_lmdb(image_path, lmdb_path):
    map_size = 10 << 40
    #print(lmdb_path)
    env = lmdb.open(lmdb_path, map_size=map_size)

    map_size = 10 << 40

    with env.begin(write=True) as txn:
        for image_fp in glob.iglob(os.path.join(image_path, '**', '*.jpg'), recursive=True):
            #image = cv2.imread(image_fp, -1)
            #image = pickle.dumps(image)
            image = open(image_fp, 'rb').read()
            image_name = os.path.basename(image_fp)
            txn.put(image_name.encode(), image)

def crop_patches(lmdb_folder, save_folder):
    env = lmdb.open(lmdb_folder, map_size=1099511627776, readonly=True, lock=False)
    with env.begin(write=False) as txn:
        image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
        #count = 0
        for name in image_names:
            data = txn.get(name)
            data = pickle.loads(data)
            mask = data['mask']
            image = data['image']

            for label in range(5):
                cls_mask = mask[:, :, label]
                insts = np.unique(cls_mask)[1:]
                for inst in insts:
                    #cls_mask[cls_mask == ]
                    row, col = np.asarray(cls_mask == inst).nonzero()
                    rmin = row.min()
                    rmax = row.max()
                    cmin = col.min()
                    cmax = col.max()
                    rmin, cmin, rmax, cmax = pad_patch(rmin, cmin, rmax, cmax, 78)

                    image_cp = image.copy()
                    bg_mask = mask[:, :, 5].astype('bool')
                    inst_mask = mask[:, :, label].astype('bool')
                    final_mask = inst_mask | bg_mask
                    image_cp[~final_mask] = 0

                    #image = cv2.rectangle(image, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

                    patch = image_cp[rmin : rmax, cmin : cmax]
                    image_name = '{}_{}_{}.jpg'.format(pickle.loads(name), label, inst)
                    cv2.imwrite(os.path.join(save_folder, image_name), image_cp)



class Prostate(Dataset):
    def __init__(self, path, transforms=None):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        self.num_classes = 6

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))

        self.transforms = transforms

        #self.image_size = image_size
        #self.mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        #self.mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        self.mean = [0.72369437, 0.44910724, 0.68094617] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        #self.std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr
        self.std = [0.17274064, 0.20472058, 0.20758244] # Expanuke bgr

        #import random
        #self.image_names = random.sample(self.image_names, k=int(len(self.image_names) / 5))
        print(len(self.image_names))

        #from torchvision import transforms
        #train_transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomVerticalFlip(),
        #    #transforms.RandomRotation(25),
        #    transforms.RandomResizedCrop(64, scale=(0.9, 1.1)),
        #    transforms.RandomApply(torch.nn.ModuleList([
        #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        #    ]), p=0.3),
        #    #transforms.ToTensor(),
        #    #transforms.Normalize(mean, std)
        #])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image = txn.get(name)
            image = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(image, -1)
            #image = pickle.loads(image)
            label = int(name.decode().split('.')[0][-1])

            if self.transforms:
                #image = self.transforms(image.astype('uint8'))
                image = self.transforms(image)

            return image, label

class PanNukEx(Dataset):
    def __init__(self, path, transforms=None):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        self.num_classes = 5

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))

        self.transforms = transforms

        #self.image_size = image_size
        #self.mean = (0.7862793912386359, 0.6027306811087783, 0.7336620786688793) #bgr
        self.mean = [0.73646324, 0.56556627, 0.70180897] # Expanuke bgr
        #self.std = (0.2111620715800869, 0.24114924152086661, 0.23603441662670357)
        self.std = [0.18869222, 0.21968669, 0.17277594] # Expanuke bgr

        #from torchvision import transforms
        #train_transforms = transforms.Compose([
        #    transforms.ToPILImage(),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomVerticalFlip(),
        #    #transforms.RandomRotation(25),
        #    transforms.RandomResizedCrop(64, scale=(0.9, 1.1)),
        #    transforms.RandomApply(torch.nn.ModuleList([
        #            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        #    ]), p=0.3),
        #    #transforms.ToTensor(),
        #    #transforms.Normalize(mean, std)
        #])
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image = txn.get(name)
            image = pickle.loads(image)
            label = int(name.decode().split('_')[1])

            if self.transforms:
                #image = self.transforms(image.astype('uint8'))
                image = self.transforms(image)

            return image, label

#lmdb_path = '/data/smb/syh/colon_dataset/PanukeEx/lmdb_data'
#lmdb_path = '/data/smb/syh/colon_dataset/PanukeEx/lmdb_data_crop'
##write_lmdb('patches', lmdb_path)
#dataset = PanNukEx(lmdb_path)
##print(len(dataset))
#for image, label in dataset:
#    print(label)
#crop_patches(lmdb_path, 'patches')


#from utils import compute_mean_std
##
#import random
#a = random.sample()
#dataset = Prostate('/data/smb/syh/PycharmProjects/CGC-Net/data_baiyu/TCGA_Prostate/Images/Patches')
##
#print(compute_mean_std(dataset))
