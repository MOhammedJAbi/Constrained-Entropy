#!/usr/bin/env python3.6
import os
import argparse
import warnings
from pathlib import Path
from operator import itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, List, Tuple

import medicalDataLoader

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

from networks import weights_init
from dataloader import get_loaders
from utils import map_
from utils import dice_coef, dice_batch, save_images, tqdm_
from utils import probs2one_hot, probs2class

from UNet import *

import matplotlib.pyplot as plt

def myloader (args):
    if args.dataset == "ISLES":
        training_dir = './Data/ISLES_TRAINING_png'
        transform = transforms.Compose([transforms.ToTensor()])
        mask_transform = transforms.Compose([transforms.ToTensor()])
        train_set = medicalDataLoader.MedicalImageDataset('train',training_dir,transform=transform,mask_transform=mask_transform,augment=False,equalize=False)
        val_set = medicalDataLoader.MedicalImageDataset('val', training_dir,transform=transform,mask_transform=mask_transform,equalize=False)
        train_loader = DataLoader(train_set,batch_size=args.batch_size,num_workers=5,shuffle=True)
        val_loader = DataLoader(val_set,batch_size=args.batch_size,num_workers=5,shuffle=False)
        training_length = len(train_set)
        val_length = len(val_set)
        num_classes = 2
        return train_loader, val_loader, training_length, val_length, num_classes
    else:
        print("The dataset is not supported.")
        raise NotImplementedError
    
def crossEntropy_f(probs,target):
    log_p = torch.log(probs + 1e-10) #.type(torch.LongTensor)
    loss = - torch.einsum("bcwh,bcwh->", [target, log_p])
    loss /= (torch.sum(target)+ 1e-10)
    return loss

def kl(target,p):
    # compute KL divergence between p and q
    return torch.sum(target * torch.log((target + 1e-8) / (p + 1e-8)))/(torch.sum(target)+ 1e-10)


def do_epoch(mode: str, args, net, device, use_cuda, loader,  optimizer, num_classes, epoch):
    
    
    totalImages = len(loader)
    
    if mode == "train":
        net.train()
        desc = f">> Training   ({epoch})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epoch})"
    
    total_iteration, total_images = len(loader), len(loader.dataset)
    all_dices: Tensor = torch.zeros((total_images, num_classes), dtype=eval(args.dtype), device=device)
    batch_dices: Tensor = torch.zeros((total_iteration, num_classes), dtype=eval(args.dtype), device=device)
    loss_log: Tensor = torch.zeros((total_images), dtype=eval(args.dtype), device=device)
    entropy_log: Tensor = torch.zeros((total_images), dtype=eval(args.dtype), device=device)
    KL_log: Tensor = torch.zeros((total_images), dtype=eval(args.dtype), device=device)
    
    tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
    done: int = 0
 
    for j, data in tq_iter:
        
        image_f,image_i,image_d,image_o,image_w,image_c, labels, img_names = data
        #image_f=image_f.type(torch.FloatTensor)/65535.
        #image_f = image_f.type(torch.FloatTensor)/65535.
        #image_i = image_i.type(torch.FloatTensor)/65535.
        #image_d = image_d.type(torch.FloatTensor)/65535.
        #image_o = image_o.type(torch.FloatTensor)/65535.
        #image_w = image_w.type(torch.FloatTensor)/65535.
        #image_c = image_c.type(torch.FloatTensor)/65535.
        MRI: Tensor = torch.zeros((1,6,image_f.size()[2], image_f.size()[3]), dtype=eval(args.dtype))
        MRI = torch.cat((image_f,image_i,image_d,image_o,image_w,image_c),dim=1)
        MRI = MRI.type(torch.FloatTensor)/65535.0 #.type(eval(args.dtype)) #.type(torch.FloatTensor)
        targets = torch.cat((1-labels,labels),dim=1) #.type(torch.LongTensor)
        B = len(image_f)
        #print(type(labels))
        #MRI = torch.cat((image_f,image_i,image_d,image_w),dim=1)
        if use_cuda:
            MRI, targets = MRI.to(device), targets.to(device)
        
        # forward
        outputs = net(MRI)
        pred_probs = F.softmax(outputs, dim=1)
        predicted_mask = probs2one_hot(pred_probs)
        
        entropy =  crossEntropy_f(pred_probs, targets)
        
        pred_probs_aver: Tensor = torch.sum(pred_probs, dim=(2,3))
        pred_probs_aver = pred_probs_aver/torch.sum(targets).float()
        target_aver: Tensor = torch.sum(targets, dim=(2,3)).float()
        target_aver = target_aver/torch.sum(targets).float()
        KL_loss = args.lam*kl(target_aver, pred_probs_aver)

        loss = entropy+ KL_loss
        
        if mode == "train":
            # zero the parameter gradients8544
            optimizer.zero_grad()
            # backward + optimize
            loss.backward()
            optimizer.step()

        
        # Compute and log metrics
        dices: Tensor = dice_coef(predicted_mask.detach(), targets.type(torch.cuda.IntTensor).detach())
        batch_dice: Tensor = dice_batch(predicted_mask.detach(), targets.type(torch.cuda.IntTensor).detach())
        assert batch_dice.shape == (num_classes,) and dices.shape == (B, num_classes), (batch_dice.shape, dices.shape, B, num_classes)

        sm_slice = slice(done, done + B)  # Values only for current batch
        all_dices[sm_slice, ...] = dices
        entropy_log[sm_slice] = entropy.detach()
        loss_log[sm_slice] = loss.detach()
        KL_log[sm_slice] = KL_loss.detach()
        batch_dices[j] = batch_dice
        
        # Logging
        big_slice = slice(0, done + B)  # Value for current and previous batches
        stat_dict = {"dice": all_dices[big_slice, -1].mean(),
                     "total loss": loss_log[big_slice].mean(),
                     "entropy loss": entropy_log[big_slice].mean(),
                     "KL loss": KL_log[big_slice].mean(),
                     "b dice": batch_dices[:j + 1, -1].mean()}
        nice_dict = {k: f"{v:.4f}" for (k, v) in stat_dict.items()}

        done += B
        tq_iter.set_postfix(nice_dict)
    
    return loss_log, entropy_log, KL_log, all_dices, batch_dices


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', '-b', default=1, type=int, help='size of the batch during training')
    parser.add_argument('--n_epoch', type=int, help='number of epoches when maximizing', default=50)
    parser.add_argument('--dataset', type=str, help='which dataset to use', default='ISLES')
    parser.add_argument('--network', type=str, help='which network to use', default='ENet')
    parser.add_argument('--lam', type=float, help='trade-off parameter for entropy and KL', default=0.)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    args = parser.parse_args()
    
    # create file and directory
    if not os.path.exists("./results"):
            os.makedirs("./results")
    
    #with open("./results/metrics_for_lambda_"+str(args.lam)+".csv", "w") as my_empty_csv:
    #    pass  
    
            
    savedir = "./results"
    # Use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Training
    print('==> loading data..')
    train_loader, val_loader, training_length, val_length, num_classes = myloader(args)
    
    # Network
    net_class = getattr(__import__('networks'), args.network)
    net = net_class(6, num_classes,eval(args.dtype)).type(eval(args.dtype))
    net.apply(weights_init)
    #net = UNetG(6, 32, num_classes)
    if use_cuda:
        net.to(device)
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', patience=4, verbose=True,factor=10 ** -0.5)
    
    n_tra: int = len(train_loader.dataset)  # Number of images in dataset
    l_tra: int = len(train_loader)  # Number of iteration per epoch: different if batch_size > 1
    n_val: int = len(val_loader.dataset)
    l_val: int = len(val_loader)

    best_dice: Tensor = torch.zeros(1).to(device)
    metrics = {"val_dice": torch.zeros((args.n_epoch, n_val, num_classes), device=device),
               "val_batch_dice": torch.zeros((args.n_epoch, l_val, num_classes), device=device),
               "val_loss": torch.zeros((args.n_epoch, n_val), device=device),
               "val_entropy": torch.zeros((args.n_epoch, n_val), device=device),
               "val_KL": torch.zeros((args.n_epoch, n_val), device=device),
               "tra_dice": torch.zeros((args.n_epoch, n_tra, num_classes), device=device),
               "tra_batch_dice": torch.zeros((args.n_epoch, l_tra, num_classes), device=device),
               "tra_loss": torch.zeros((args.n_epoch, n_tra), device=device),
               "tra_entropy": torch.zeros((args.n_epoch, n_tra), device=device),
               "tra_KL": torch.zeros((args.n_epoch, n_tra), device=device)}
    
    #run
    for epoch in range(args.n_epoch):
        
        tra_loss, tra_entropy, tra_KL, tra_dice, tra_batch_dice = do_epoch("train", args, net, device, use_cuda, train_loader, optimizer,num_classes,epoch)
        
        with torch.no_grad():
            val_loss, val_entropy, val_KL, val_dice, val_batch_dice = do_epoch("val", args, net, device, use_cuda, val_loader, optimizer,num_classes,epoch)
        
        
        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][epoch].shape == eval(k).shape, (metrics[k][epoch].shape, eval(k).shape)
            metrics[k][epoch] = eval(k)

        #for k, e in metrics.items():
        #    np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=1).cpu().numpy(),
                           "val_loss": metrics["val_loss"].mean(dim=1).cpu().numpy(),
                           "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "tra_batch_dice": metrics["tra_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy(),
                           "val_batch_dice": metrics["val_batch_dice"][:, :, -1].mean(dim=1).cpu().numpy()})
        df.to_csv(Path(savedir, "metrics_for_lambda_"+str(args.lam)+".csv"), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_dice: Tensor = val_dice[:, -1].mean()
        if current_dice > best_dice:
            best_dice = current_dice
            with open(Path(savedir, "best_epoch_for_lambda_"+str(args.lam)+".txt"), 'w') as f:
                f.write(str(epoch))
            best_folder = Path(savedir, "best_epoch")
            if best_folder.exists():
                rmtree(best_folder)
#            copytree(Path(savedir, f"iter{epoch:03d}"), Path(best_folder))
            torch.save(net, Path(savedir, "best.pkl"))
            
            
        
    
    
    
 