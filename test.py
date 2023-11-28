#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:23:34 2020
@author: zzl
"""
import torch
import torchvision.transforms as transforms
from networks import net_G
from PIL import Image
import os
import numpy as np

img_path = 'Z:/AML/SwissSamplesTiles_forZelin2023/1152_tp6/'
save_path = 'Z:/AML/SwissSamplesTiles_forZelin2023/1152_tp6_pred/'

if not os.path.exists(save_path):
   os.makedirs(save_path) 
    
net_g = net_G()
net_g.load_state_dict(torch.load('netG_model_epoch_30.pth'))
net_g.eval()
net_g.cuda()

transform_list = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img_list = os.listdir(img_path)
for i in range(len(img_list)):
    img_name = img_list[i]
    Img = Image.open(img_path+img_name).convert('RGB')
    Input = transform_list(Img).unsqueeze(0).cuda()
    with torch.no_grad():
         pred = net_g(Input)
    out_img = pred.detach().squeeze(0).cpu().float().numpy()
    z = np.transpose(out_img, (1, 2, 0))
    image_numpy = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    result = image_numpy.copy()
    save_Img = Image.fromarray(result)
    save_Img.save(save_path+img_name[:-4]+'.png')
    print('{}/{} done'.format(i+1,len(img_list)))
    
    
    
    
    
    
    
    
