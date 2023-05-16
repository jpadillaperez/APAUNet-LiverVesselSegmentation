# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from medpy import metric
from config import train_config 

class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False, sigmoid=True, n_classes=1):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if sigmoid:
            inputs = torch.nn.Sigmoid()(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            #print("Dice Loss: " + str(dice_vessel.float().cpu().detach().numpy()))
            print("Dice Score: " + str(1 - dice_loss.float().cpu().detach().numpy()))
            loss += dice_loss
        return loss
    

class DiceLossWeighted(nn.Module):
    def __init__(self, weight=None):
        super(DiceLossWeighted, self).__init__()
        self.weight = weight

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False, sigmoid=True, n_classes=1):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if sigmoid:
            inputs = torch.nn.Sigmoid()(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, n_classes):
            dice_vessel = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice_vessel.item())
            #print("Dice Loss Vessel: " + str(dice_vessel.float().cpu().detach().numpy()))
            print("Dice Score Vessel: " + str(1 - dice_vessel.float().cpu().detach().numpy()))

            ##Try implementing separated vessel
            dice_background = self._dice_loss(1 + inputs[:, i], 1 + target[:, i])
            #print("Dice Loss Background: " + str(dice_background.float().cpu().detach().numpy()))
            print("Dice Score Background: " + str(1 - dice_background.float().cpu().detach().numpy()))
            
            if(self.weight == None):
                loss += dice_vessel + dice_background
            else:
                loss += dice_vessel * self.weight[0] + dice_background * self.weight[1]

            #print("Overall Dice Loss: " + str(loss.float().cpu().detach().numpy()))
            print("Overall Dice Score: " + str(loss.float().cpu().detach().numpy()))
        return loss
    
class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        if train_config["gamma"] != None or train_config["alpha"] != None:
            self.gamma = train_config["gamma"]
            self.alpha = train_config["alpha"]
            print("Binary loss with gamma: " + str(train_config["gamma"]) + "and alpha: " + str(train_config["alpha"]))
        else:
            self.gamma = gamma
            self.alpha = alpha
            print("Binary loss with gamma: " + str(self.gamma) + "and alpha: " + str(self.alpha))

        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = - pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = - self.alpha * neg_weight * F.logsigmoid(- output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss
    
class DiceFocalLoss(nn.Module):
    def __init__(self, beta = 10):
        super(DiceFocalLoss, self).__init__()
        self.focal_loss = BinaryFocalLoss()
        self.dice_loss = DiceLoss()
        self.beta = beta

    def forward(self, inputs, target):
        return (self.beta * self.focal_loss(inputs, target)) - torch.log(self.dice_loss(inputs, target))


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def HD95(outputs, targets):
    output = 0 * outputs[:,0] + 1 * outputs[:,1] + 2 * outputs[:,2]
    output = output.unsqueeze(1)
    hd95 = metric.binary.hd95(output.detach().cpu().numpy(), targets.detach().cpu().numpy())
    return hd95

def is_image(fname):
    return fname.find('data.npy') != -1
