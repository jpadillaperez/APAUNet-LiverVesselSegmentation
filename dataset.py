# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import numpy as np

def padding(sample, target, input_shape):
    HWD = np.array(input_shape)
    hwd = np.array(target.shape)
    tmp = np.clip((HWD - hwd) / 2, 0, None)
    rh, rw, rd = np.floor(tmp).astype(int)
    lh, lw, ld = np.ceil(tmp).astype(int)

    if sample.ndim == 3:
        sample = np.pad(sample, ((rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    else:
        sample = np.pad(sample, ((0,0), (rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    return sample, target

def random_crop(sample, target, input_shape):
    H, W, D = input_shape
    h, w, d = target.shape
    
    x = np.random.randint(0, h - H + 1)
    y = np.random.randint(0, w - W + 1)
    z = np.random.randint(0, d - D + 1)

    if sample.ndim == 3:
        return sample[x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]
    else:
        return sample[:, x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]

def bounding_crop(sample, target, input_shape):
    H, W, D = input_shape
    source_shape = list(target.shape)
    xyz = list(np.where(target > 0))

    lower_bound = []
    upper_bound = []
    for A, a, b in zip(xyz, source_shape, input_shape):

        if len(A) == 0:
            print("Error! Size of the vessel ROI is smaller than the input shape. Ignoring...")
            return [],[]

        lb = max(np.min(A), 0) # type: ignore
        ub = min(np.max(A)-b+1, a-b+1)

        if ub <= lb:
            lb = max(np.max(A) - b, 0)
            ub = min(np.min(A), a-b+1) # type: ignore
        
        lower_bound.append(lb)
        upper_bound.append(ub)
    x, y, z = np.random.randint(lower_bound, upper_bound)

    if sample.ndim == 3:
        return sample[x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]
    else:
        return sample[:, x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]

def random_mirror(sample, target, prob=0.5):
    p = np.random.uniform(size=3)
    axis = tuple(np.where(p < prob)[0])
    sample = np.flip(sample, axis)
    target = np.flip(target, axis)
    return sample, target

def brightness(sample, target):    
    sample_new = np.zeros(sample.shape)
    for c in range(sample.shape[-1]):
        im = sample[:,:,:,c]        
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        sample_new[:,:,:,c] = im_new 
    
    return sample_new, target

def totensor(data):
    return torch.from_numpy(np.ascontiguousarray(data))

def mp_get_datas(data_dir, image_ids):

    def get_item(idx):
        output = np.load(os.path.join(data_dir, idx))
        target = output[1,:,:,:]
        sample = output[0,:,:,:]
        target[target != 1] = 0
        return (sample, target)

    data = []
    for id in image_ids:
        (sample, target) = get_item(id)
        data.append([sample, target])

    return data

def mp_get_batch(data, idxs, input_shape, aug='random'):
    crop = random_crop if aug == 'random' else bounding_crop

    def batch_and_aug(idx):
        sample, target = data[idx]
        sample, target = padding(np.squeeze(sample), np.squeeze(target), input_shape)
        sample, target = crop(sample, target, input_shape)
        if ((len(sample) == 0) & (len(target) == 0)):
            return [],[]
        sample, target = random_mirror(sample, target)

        sample = totensor(sample)
        target = totensor(target)
        if sample.dim() == 3:
            sample.unsqueeze_(0)
        if target.dim() == 3:
            target.unsqueeze_(0)
        sample.unsqueeze_(0)
        target.unsqueeze_(0)
        return sample, target

    batch = []
    for id in idxs:
        (sample, target) = batch_and_aug(id)
        if ((len(sample) == 0) & (len(target) == 0)):
           return [],[]
        batch.append([sample, target])

    samples, targets = zip(*batch)
    samples = torch.cat(samples)
    targets = torch.cat(targets)

    return samples, targets
