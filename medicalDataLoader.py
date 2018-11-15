from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        
        train_inn_path = os.path.join(root, 'train', 'CT')
        train_dwi_path = os.path.join(root, 'train', 'CT_4DPWI')
        train_cbf_path = os.path.join(root, 'train', 'CT_CBF')
        train_fat_path = os.path.join(root, 'train', 'CT_CBV')
        train_opp_path = os.path.join(root, 'train', 'CT_MTT')
        train_wat_path = os.path.join(root, 'train', 'CT_Tmax')
        train_mask_path = os.path.join(root, 'train', 'OT')
        
        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_dwi = os.listdir(train_dwi_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        images_cbf = os.listdir(train_cbf_path)
        
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_dwi.sort()
        images_opp.sort()
        images_wat.sort()
        images_cbf.sort()
        labels.sort()

        for it_f,it_i,it_d,it_o,it_w, it_c, it_gt in zip(images_fat,images_inn,images_dwi,images_opp,images_wat,images_cbf,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_dwi_path, it_d),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_cbf_path, it_c),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
            
    elif mode == 'val':
        train_inn_path = os.path.join(root, 'val', 'CT')
        train_dwi_path = os.path.join(root, 'val', 'CT_4DPWI')
        train_cbf_path = os.path.join(root, 'val', 'CT_CBF')
        train_fat_path = os.path.join(root, 'val', 'CT_CBV')
        train_opp_path = os.path.join(root, 'val', 'CT_MTT')
        train_wat_path = os.path.join(root, 'val', 'CT_Tmax')
        train_mask_path = os.path.join(root, 'val', 'OT')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_dwi = os.listdir(train_dwi_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        images_cbf = os.listdir(train_cbf_path)
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_dwi.sort()
        images_opp.sort()
        images_wat.sort()
        images_cbf.sort()
        labels.sort()

        for it_f,it_i,it_d,it_o,it_w, it_c, it_gt in zip(images_fat,images_inn,images_dwi,images_opp,images_wat,images_cbf,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_dwi_path, it_d),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_cbf_path, it_c),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)
    else:
        train_fat_path = os.path.join(root, 'test', 'CBV')
        train_inn_path = os.path.join(root, 'test', 'CT')
        train_dwi_path = os.path.join(root, 'test', 'DWI')
        train_opp_path = os.path.join(root, 'test', 'MTT')
        train_wat_path = os.path.join(root, 'test', 'Tmax')
        train_mask_path = os.path.join(root, 'test', 'GT')

        images_fat = os.listdir(train_fat_path)
        images_inn = os.listdir(train_inn_path)
        images_dwi = os.listdir(train_dwi_path)
        images_opp = os.listdir(train_opp_path)
        images_wat = os.listdir(train_wat_path)
        labels = os.listdir(train_mask_path)

        images_fat.sort()
        images_inn.sort()
        images_dwi.sort()
        images_opp.sort()
        images_wat.sort()
        labels.sort()

        for it_f,it_i,it_d,it_o,it_w, it_gt in zip(images_fat,images_inn,images_dwi,images_opp,images_wat,labels):
            item = (os.path.join(train_fat_path, it_f),
                    os.path.join(train_inn_path, it_i),
                    os.path.join(train_dwi_path, it_d),
                    os.path.join(train_opp_path, it_o),
                    os.path.join(train_wat_path, it_w),
                    os.path.join(train_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        fat_path, inn_path,dwi_path,opp_path,wat_path,c_path,mask_path = self.imgs[index]
        # print("{} and {}".format(img_path,mask_path))
        #img = Image.open(img_path)  # .convert('RGB')
        #mask = Image.open(mask_path)  # .convert('RGB')
        img_f = Image.open(fat_path)#.convert('L')
        img_i = Image.open(inn_path)#.convert('L')
        img_d = Image.open(dwi_path)#.convert('L')
        img_o = Image.open(opp_path)#.convert('L')
        img_w = Image.open(wat_path)#.convert('L')
        img_c = Image.open(c_path)
        mask = Image.open(mask_path).convert('L')
        
        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img_f = self.transform(img_f)
            img_i = self.transform(img_i)
            img_d = self.transform(img_d)
            img_o = self.transform(img_o)
            img_w = self.transform(img_w)
            img_c = self.transform(img_c)
            mask = self.mask_transform(mask)

        return [img_f,img_i,img_d,img_o,img_w,img_c, mask, fat_path]
