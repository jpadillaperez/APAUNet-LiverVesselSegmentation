# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

train_config = {
        "model_name":   "APAUNet",
        "dataset":      "Task08_HepaticVessel",
        "data_path":    "/data/jorge_perez/Task08_HepaticVessel_preprocessed/08_3d/Train",
        "scheduler":    "StepLR",
        "criterion":    "Dice",
        "optimizer":    "Adam",
        "gamma":        None, #For Focal Loss
        "alpha":        None, #For Focal Loss
        "lr":           0.0001,
        "epochs":       600,
        "val_interval": 10,
        "batch_size":   2,
        "in_ch":        1,
        "class_num":    1,
        "val_num":      50,
        "input_shape":  (80, 80, 80),
        "resume":       None,
        "use_cuda":     True,
        "debug":        False
}

test_config = {
        "model_name":   "APAUNet",
        "dataset":     "Task08_HepaticVessel",
        "data_dir":    "/data/jorge_perez/Task08_HepaticVessel_preprocessed/08_3d/Test",
        "in_ch":        1,
        "class_num":    1,
        "batch_size":   2,
        "input_shape":  (80, 80, 80),
        "resume":       "/data/jorge_perez/Models/model_APAUNetHepaticVessel_epoch200_DICE_107817.pth",
        "use_cuda":     True
        }