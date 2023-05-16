import os
import torch
import numpy as np
from utils import DiceLoss, is_image
from build_model import build_model
from config import test_config
import SimpleITK as sitk
from polyaxon_client.tracking import get_outputs_path
from dataset import mp_get_datas, mp_get_batch

#from monai.inferers.utils import sliding_window_inference

# ------ test data config ------
data_dir    = test_config["data_dir"]
ids         = os.listdir(data_dir)
test_ids         = list(filter(is_image, ids))
print('Test', test_ids)
test_data  = mp_get_datas(data_dir, test_ids, test_config["dataset"]) # type: ignore
test_list  = list(range(len(test_ids)))

# ------ extra parameters ------
input_size  = test_config["input_shape"]
criterion = DiceLoss()

# ------ load model checkpoint ------
model = build_model(test_config["model_name"], test_config["in_ch"], test_config['class_num'])
model.load_state_dict(torch.load(test_config['resume'])['state_dict'])
if test_config["use_cuda"]:
    model = model.cuda()

#Tester
model.eval()
loss_list   = []
for i in range(0, len(test_list), 2):
    with torch.no_grad():
        if i + test_config["batch_size"] > len(test_list):
            break

        #The bounding crop algorithm to find the crops of 80,80,80 would be changed by a liver segmentation coming from a different architecture in the future.
        sample_a, target_a = mp_get_batch(test_data, test_list[i:i+test_config["batch_size"]//2], test_config["input_shape"], aug='bounding')
        sample_b, target_b = mp_get_batch(test_data, test_list[i+test_config["batch_size"]//2:i+test_config["batch_size"]], test_config["input_shape"], aug='bounding')

        if ((len(sample_a) == 0) or (len(sample_b) == 0)):
            continue

        inputs  = torch.cat((sample_a, sample_b), 0)
        targets = torch.cat((target_a, target_b), 0)

        if test_config["use_cuda"]:
            inputs  = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        loss    = criterion(outputs, targets)
        loss_list.append(loss.item())

        outputs = torch.nn.Sigmoid()(outputs)

        #Saving the resulting volumes
        sitk.WriteImage(sitk.GetImageFromArray((outputs[0][0]>0.5).float().cpu().detach().numpy()), os.path.join(get_outputs_path(), 'outputs_batch' + str(i) + '.nii.gz')) # type: ignore
        sitk.WriteImage(sitk.GetImageFromArray(targets[0][0].float().cpu().numpy()), os.path.join(get_outputs_path(), 'targets_batch' + str(i) + '.nii.gz')) # type: ignore
        
        del inputs, targets, outputs
       
'''
#Alternative test with sliding window inference

        output = sliding_window_inference(inputs = sample, overlap = 0, roi_size=input_size, sw_batch_size=1, predictor=model, sw_device='cuda', device="cpu", progress= True)
        output = output.cuda()
        loss    = criterion(outputs, targets)
        loss_list.append(loss.item())
        output = torch.nn.Sigmoid()(output)
        sitk.WriteImage(sitk.GetImageFromArray((output[0][0]>0.5).float().cpu().detach().numpy()), os.path.join(get_outputs_path(), test_ids[i][:-4] + '.nii.gz')) # type: ignore
        sitk.WriteImage(sitk.GetImageFromArray((target[0][0]).float().cpu().detach().numpy()), os.path.join(get_outputs_path(), test_ids[i][:-4] + '_target.nii.gz')) # type: ignore
        del sample, target, output

'''

print("Average dice score: ")
print(1 - np.mean(loss_list))