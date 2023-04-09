#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:59:38 2021

@author: dawood
"""

import torch; print(torch.__version__)
import torch; print(torch.version.cuda)

import numpy as np
import os
from scipy.spatial import Delaunay, KDTree
from collections import defaultdict
import torch
from torch_geometric.data import Data
from torch.autograd import Variable
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.neighbors import KDTree as sKDTree
from tqdm import tqdm
import pickle
import pandas as pd
import os

def mkdirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def toTensor(v,dtype = torch.float,requires_grad = True): 
    device = 'cpu'   
    return (Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad)).to(device)

def connectClusters(Cc,dthresh = 3000):
    tess = Delaunay(Cc)
    neighbors = defaultdict(set)
    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    nx = neighbors    
    W = np.zeros((Cc.shape[0],Cc.shape[0]))
    for n in nx:
        nx[n] = np.array(list(nx[n]),dtype = np.int)
        nx[n] = nx[n][KDTree(Cc[nx[n],:]).query_ball_point(Cc[n],r = dthresh)]
        W[n,nx[n]] = 1.0
        W[nx[n],n] = 1.0        
    return W # neighbors of each cluster and an affinity matrix

def toGeometric(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def build_graph(d,label=1):
    x, y, F = d['x_patch'], d['y_patch'], d['feat']
    #import pdb; pdb.set_trace()
    C = np.asarray(np.vstack((x, y)).T, dtype=np.int)
    W = connectClusters(C, dthresh=4000) # dthresh: threshold value for connecting patches
    G = toGeometric(F, W, y=label)
    G.coords = toTensor(C, requires_grad=False)
    cpu = torch.device('cpu')
    G.to(cpu)
    with open(ofile, 'wb') as f:
          pickle.dump(G, f)


bdir = '/data/PanCancer/HTEX_repo'
DATA_DIR = f'{bdir}/data'
patch_size = (512,512)

TAG = 'x'.join(map(str,patch_size))

Repr = 'ShuffleNet'

batch_size = 512

FEATURES_DIR = f'{DATA_DIR}/Features/{TAG}/{Repr}/'               
GRAPHS_DIR = f'{DATA_DIR}/Graphs/{TAG}/{Repr}/' 

 # path to where to dump the Graphs

mkdirs(GRAPHS_DIR)

ex_list = [] # add problamatic files
for filename in os.listdir(FEATURES_DIR):
    if filename.endswith(".npz"):  
        ofile = os.path.join(GRAPHS_DIR, filename[:23] + '.pkl')
        if os.path.isfile(ofile):
            continue
        d = np.load(f'{FEATURES_DIR}{filename}', allow_pickle=True)
        try:
            build_graph(d)
            print('Finished:', filename)  
        except:
            print("problametic feature extraction")
            ex_list.append(filename)
            print(ex_list)
            np.savez(f'{GRAPHS_DIR}/execp.npz',ex_list)
