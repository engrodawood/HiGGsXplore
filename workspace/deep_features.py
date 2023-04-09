#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:38:17 2020
Extract features using a pretrained model (ShuffleNet)
@author: Muhammad Dawood
"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import torch.nn as nn
from skimage.io import imread

USE_CUDA = torch.cuda.is_available()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA] 
device = torch.device(device)
def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

class FeatureExtractor():
    def __init__(self,
                repo='pytorch/vision:v0.10.0',
                model_name = 'shufflenet_v2_x1_0',
                pretrained=True
                ):
        self.repo = repo
        self.model_name = model_name
        self.pretrained = pretrained

    def instantiate(self):
        model = torch.hub.load(self.repo,
                                    self.model_name, 
                                    pretrained=self.pretrained
                                    )
        # if the model is missing adaptive pooling layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        randImg = torch.rand((1,3,224,224))

        if len(model(randImg).shape)>2:
            self.model = torch.nn.Sequential(*list(model.children()),nn.AdaptiveAvgPool2d(1))
        return self.model


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, patches_path,transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensor = patches_path
        self.transform = transform
        
    def __getitem__(self, index):
        x = imread(self.tensor[index])[:,:,:3]

        if self.transform:
            x = self.transform(x)
        return x, self.tensor[index]
    def __len__(self):
        return len(self.tensor)
    
def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

bdir = '/data/PanCancer/HTEX_repo'
DATA_DIR = f'{bdir}/data'
patch_size = (512,512)

TAG = 'x'.join(map(str,patch_size))
TILES_DIR = f'{DATA_DIR}/Patches/{TAG}'

Repr = 'ShuffleNet'

batch_size = 512

FEATURES_DIR = f'{DATA_DIR}/Features/{TAG}/{Repr}/'
# making directories for Features if not there

mkdirs(FEATURES_DIR)

# Creating Feature Extractor Object
FE = FeatureExtractor().instantiate()

for param in FE.parameters():
    param.requires_grad = False

FE.eval()

FE.to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def file2coord(flist):
    xl,yl=[],[]
    for file in flist:
        patch_name = file.split('/')[-1]
        coords = patch_name.split('.')[0].split('_')
        xl.append(int(coords[0]))
        yl.append(int(coords[1]))
    return xl,yl

for patient in os.listdir(TILES_DIR):
    print(patient, "Extracting features....")
    x_patch=[]
    y_patch = []
    patches_names = []
    deep_features = []

    patches_path = os.path.join(TILES_DIR,patient)
    patches_list = [f'{patches_path}/{f}' for f in os.listdir(patches_path)]
    
    if len(patches_list)==0:
        continue
    ofile = os.path.join(FEATURES_DIR, patient +'_'+ Repr + '.npz')
    if os.path.isfile(ofile): 
        continue
    
    dataset = CustomTensorDataset(patches_list,transform=preprocess)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,pin_memory=True,num_workers=10,shuffle=False)
    
    for x,filename in tqdm(test_loader):
        output = FE(x.to(device)).squeeze().detach().cpu().numpy()
        xl,yl = file2coord(filename)
        x_patch.extend(xl)
        y_patch.extend(yl)
        if x.shape[0]==1:
            deep_features.append(output)
            patches_names.append(filename)
        else:
            deep_features.extend(output)
            patches_names.extend(filename)
    print(np.shape(x_patch),np.shape(y_patch),np.shape(deep_features))
    
    # Saving features and Patch Coordinates
    np.savez(ofile,feat=deep_features,x_patch=x_patch,y_patch=y_patch,pid=patient)

        

    
