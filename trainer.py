# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import random
import torch
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import  GradScaler
from utils import ModelEma
from dataset import mp_get_batch
from datetime import datetime
import SimpleITK as sitk

from polyaxon_client.tracking import get_outputs_path

class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler, train_list, val_list, train_data, val_data, config, use_cuda = True, use_ema = True):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.use_cuda   = use_cuda
        self.use_ema    = use_ema
        self.train_list = train_list
        self.val_list   = val_list
        self.train_data = train_data
        self.val_data   = val_data
        self.config     = config
        self.epochs     = self.config["epochs"]
        self.best_dice  = 0 
        self.best_epoch = 0
        self.class_num = self.config["class_num"]
        if self.use_cuda and config['resume'] is None:
            self.model  = self.model.cuda()        
        if self.use_ema:
            self.ema    = ModelEma(self.model, decay=0.9998)

    def run(self):
        scaler = GradScaler()
        for epoch in range(self.epochs):
            self.train(epoch, scaler)
            if self.config["model_name"] == "APAUNet":
                if (epoch + 1) % 10 == 0:
                    print(self.model.encoder1.beta, self.model.encoder2.beta, self.model.encoder3.beta, self.model.encoder4.beta, self.model.encoder5.beta)
                    print(self.model.decoder1.beta, self.model.decoder2.beta, self.model.decoder3.beta, self.model.decoder4.beta)
            if (epoch % self.config["val_interval"] == 0):
                dice_mean = self.evaluation(epoch)
                if epoch >= 10 and self.best_dice < dice_mean:
                    self.save(dice_mean, epoch)
            if epoch % 100 == 0:
                checkpoint = {
                    'epoch': epoch, 
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }
                torch.save(checkpoint, os.path.join(get_outputs_path(), "model_" + self.config["model_name"]+ self.config["dataset"][7:]+ '_epoch' + str(epoch) + '.pth')) # type: ignore
       


    def train(self, epoch, scaler):
        self.model.train()
        Time        = []
        loss_list   = []
        random.shuffle(self.train_list)

        for i in range(0, len(self.train_list), 2):
            if i + self.config["batch_size"] > len(self.train_list):
                break

            ##Changed to have crops just in positions in which it is not empty
            sample_a, target_a = mp_get_batch(self.train_data, self.train_list[i:i+self.config["batch_size"]//2], self.config["input_shape"], aug='bounding')
            sample_b, target_b = mp_get_batch(self.train_data, self.train_list[i+self.config["batch_size"]//2:i+self.config["batch_size"]], self.config["input_shape"], aug='bounding')
            
            if ((len(sample_a) == 0) or (len(sample_b) == 0)):
                continue

            inputs  = torch.cat((sample_a, sample_b), 0)
            targets = torch.cat((target_a, target_b), 0)

            if self.use_cuda:
                inputs  = inputs.cuda()
                targets = targets.cuda()
            with autocast():
                outputs = self.model(inputs)
                
                #Changed Monai Dice by self made Dice Loss
                loss    = self.criterion(outputs, targets)

                output_t = torch.nn.Sigmoid()(outputs)
                #output_t = torch.argmax(output_t, 1) #to round to 1 or 0 instead of >0.5

                sitk.WriteImage(sitk.GetImageFromArray((output_t[0][0]>0.5).float().cpu().detach().numpy()), os.path.join(get_outputs_path(), 'outputs_batch' + str(i) +'_epoch_'+ str(epoch) +'.nii.gz')) # type: ignore
                sitk.WriteImage(sitk.GetImageFromArray(targets[0][0].float().cpu().numpy()), os.path.join(get_outputs_path(), 'targets_batch' + str(i) +'_epoch_'+ str(epoch) +'.nii.gz')) # type: ignore

            self.optimizer.zero_grad()

            #Using a scaler to increase the size of the loss
            #Using ema to improve the stability of the model
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()    
            self.ema.update(self.model)        
            loss_list.append(loss.item())

            del inputs, targets, outputs
            
        self.scheduler.step()
        print("-"*20)
        print(f"{datetime.now()} Training--epoch: {epoch+1}\t"f" lr: {self.scheduler.get_last_lr()[0]:.6f}\t"f" batch loss: {np.mean(loss_list):.6f}\t")

    def evaluation(self, epoch):
        # a trick for quick evaluate on validation set, the formal evaluate should use slide_windows_inference in MONAI.
        self.model.eval()
        dice_mean_list  = []
        dice_background_list = []
        dice_vessel_list = []
        with torch.no_grad():
            for i in range(0, len(self.val_list), 2):
                inputs, targets = mp_get_batch(self.val_data, self.val_list[i:i+2], self.config["input_shape"], aug='bounding')
                
                if ((len(inputs) == 0) or (len(targets) == 0)):
                    continue

                if self.use_cuda:
                    inputs  = inputs.cuda() # type: ignore
                    targets = targets.cuda() # type: ignore

                outputs = self.model(inputs)
                outputs = torch.nn.Sigmoid()(outputs)

                dice_vessel_score = 1 - self._dice_loss(outputs[:, 0], targets[:, 0]) # type: ignore
                dice_background_score = 1 - self._dice_loss(1 + outputs[:, 0], 1 + targets[:, 0]) # type: ignore
                dice_mean_score = dice_vessel_score * 0.75 + dice_background_score * 0.25

                dice_mean_list.append(dice_mean_score.float().cpu().detach().numpy())
                dice_background_list.append(dice_background_score.float().cpu().detach().numpy())
                dice_vessel_list.append(dice_vessel_score.float().cpu().detach().numpy())
        
        print("-"*20)
        print("EVALUATION")
        print(f"dice_average_score: {np.mean(dice_mean_list):0.4f}")
        print(f"dice_background_score: {np.mean(dice_background_list):0.6f}\t"f"dice_vessel_score: {np.mean(dice_vessel_list):0.6f}\t")
        print("-"*20)
        return np.mean(dice_mean_list)
    
    def save(self, dice_mean, epoch):
        self.best_dice = dice_mean
        self.best_epoch = epoch
        checkpoint = {
            'epoch': epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
                    }
        torch.save(checkpoint, os.path.join(get_outputs_path(), 'best_'+self.config["model_name"]+ self.config["dataset"][7:]+ '_epoch' + str(epoch) + '.pth')) # type: ignore
        print(f"best epoch: {self.best_epoch}, best dice: {self.best_dice:.4f}")
        print('Success saving model')

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss