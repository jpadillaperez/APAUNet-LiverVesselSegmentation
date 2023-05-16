# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
from monai.inferers.utils import sliding_window_inference
from config import test_config
from build_model import build_model
from polyaxon_client.tracking import get_data_paths, get_outputs_path
from utils import DiceLoss, HD95

def main():
    # ------ load model checkpoint ------
    model = build_model(test_config["model_name"], test_config["in_ch"], test_config['class_num'])
    if test_config["use_cuda"]:
        model.load_state_dict(torch.load(test_config['resume'], map_location=torch.device('cuda'))['state_dict'])
        model.cuda()
    else:
        #model.load_state_dict(torch.load("C:/Users/Jorge/Desktop/best_APAUNetHepaticVessel_epoch110.pth", map_location=torch.device('cpu'))['state_dict'])
        model.load_state_dict(torch.load(test_config['resume'], map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    # ------ test data config ------
    data_dir = test_config["data_dir"]
    input_size  = test_config["input_shape"]
    dice_loss = DiceLoss()

    # ------ inference ------
    print('inference start!')
    
    for index in tqdm(os.listdir(data_dir)):
        img = np.array(nib.load(os.path.join(data_dir, index)).get_fdata())
        img = torch.from_numpy(img)
        
        #print("Shape of input: ")
        #print(img.shape)

        img = img.to(torch.float32)
        img = torch.unsqueeze(img, 0).unsqueeze(0)
        with torch.no_grad():
            if test_config["use_cuda"]:
                img = img.cuda()
                output = sliding_window_inference(img, roi_size=input_size, sw_batch_size=1, predictor=model, sw_device='cuda', device="cpu", progress= True)
            else:
                output = sliding_window_inference(img, roi_size=input_size, sw_batch_size=1, predictor=model, sw_device='cpu', device="cpu", progress= True)
            
            #Trial refinement output
            output = torch.nn.Sigmoid()(output)
            output = torch.argmax(output, 1)
            output = (output[0]).float().cpu().numpy() ##Check from here

            #Evaluation
            dice_loss(output, target)




            
        #print("Shape of output: ")
        #print(output.shape)

        sitk.WriteImage(sitk.GetImageFromArray(output), os.path.join(get_outputs_path(), index)) # type: ignore

    print('inference over!')

if __name__ == '__main__':
    main()
