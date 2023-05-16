# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

from model.APAUNet import APAUNet

def build_model(model_name, in_ch, out_ch):
    if model_name == 'APAUNet':
        print('Loading model APAUNet!')
        return APAUNet(in_ch, out_ch)
    else:
        raise RuntimeError('Given model name not implemented!')