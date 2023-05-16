# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import random
import torch
from build_model import build_model
from utils import is_image
from dataset import mp_get_datas
from trainer import Trainer
from config import train_config 
from utils import DiceLoss, DiceFocalLoss, DiceLossWeighted, DiceBCELoss, BinaryFocalLoss

def main():    
    # -------- load data --------
    assert train_config['dataset'] in ['Task007_Pancreas', 'Task004_Liver', 'Task08_HepaticVessel'], 'Other dataset is not implemented'

    #Run on polyaxon
    if train_config["debug"]:
        train_config["data_path"] = "/data/jorge_perez/Small_train"
        train_config["val_num"] = 2
        train_config["epochs"] = 30
        print("Changed configuration to debugging mode... Ignoring val_num, epochs and data_path.")

    data_dir = train_config["data_path"]

    ids         = os.listdir(data_dir)
    ids         = list(filter(is_image, ids))

    val_ids     = random.sample(ids, train_config["val_num"])
    train_ids   = []
    for idx in ids:
        if idx in val_ids:
            continue
        train_ids.append(idx)
    print('Val', val_ids)
    print('Train', train_ids)

    val_data    = mp_get_datas(data_dir, val_ids, train_config["dataset"]) # type: ignore
    train_data  = mp_get_datas(data_dir, train_ids, train_config["dataset"]) # type: ignore
    train_list  = list(range(len(train_ids)))
    val_list    = list(range(len(val_ids)))
    print(f'Get datas finished. Train data: {len(train_list)}, Validation data: {len(val_list)}')

    # -------- load model --------
    model       = build_model(train_config["model_name"], train_config["in_ch"], train_config['class_num'])

    # -------- Loss functions --------
    if train_config["criterion"] == 'DiceFocal':
        criterion = DiceFocalLoss()
        print('---------Using DiceFocal Loss')
    elif train_config["criterion"] == 'Dice':
        criterion = DiceLoss()                      #criterion = DiceLoss(weight=[0.75, 0.25])
        print('---------Using Dice Loss')
    elif train_config["criterion"] == 'DiceWeighted':
        criterion = DiceLossWeighted()
        print('---------Using Dice Weighted Loss')
    elif train_config["criterion"] == 'DiceBCELoss':
        criterion = DiceBCELoss()
        print('---------Using Dice BCE Loss (Under development)') 
    elif train_config["criterion"] == 'BinaryFocalLoss':
        criterion = BinaryFocalLoss()
        print('---------Using Binary Focal Loss')
    else:
        raise NotImplementedError

    # -------- Optimizers --------
    if train_config["optimizer"] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_config["lr"], momentum=0.9, weight_decay=0.0001)
        print('---------Using SGD Optimizer')
    elif train_config["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], betas=(0.9, 0.99))
        print('---------Using Adam Optimizer')
    elif train_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        raise NotImplementedError

    # -------- Learning rate schedulers & Warm up tricks --------
    if train_config["scheduler"] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=350, eta_min=0.000001)
        print('---------Using CosineAnnealingLR Warmup')
    elif train_config["scheduler"] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 50, 80, 100, 120, 135], gamma=0.1)
        print('---------Using MultiStepLR Warmup')
    elif train_config["scheduler"] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.75)
    else:
        raise NotImplementedError

    # -------- Checkpoint resume ----------
    if train_config["resume"] is not None:
        print("loading saved Model...")
        checkpoint  = torch.load(train_config["resume"])
        model.load_state_dict(checkpoint['state_dict'])
        #model.to("cuda")
        model       = model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch       = checkpoint['epoch']
        print("Model successfully loaded! Current step is: ", epoch)   

    # -------- Training ----------
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, train_list, val_list, train_data, val_data, train_config, train_config["use_cuda"])
    trainer.run()

if __name__ == '__main__':
    main()
