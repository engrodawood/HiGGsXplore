'''
Importing packages
'''

from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from scipy.spatial import distance_matrix, Delaunay
import random
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from gnn import *
from pathlib import Path


if __name__ == '__main__':


    patch_size = (512,512)
    bdir = '/data/PanCancer/HTEX_repo'
    DATA_DIR = f'{bdir}/data'
    TAG = 'x'.join(map(str,patch_size))
    Repr = 'ShuffleNet'
    OUTPUT_DIR = f'{bdir}/Output/{TAG}/{Repr}/' 

    # For using Pre-trained weights please change the path to pretrained
    WPATH = f'{OUTPUT_DIR}/WEIGHTS/{TAG}'

    GRAPHS_DIR = '/data/CPTAC/GRAPHS/20x-SHUFFLE_NET'
    graphlist = [g for g in glob(os.path.join(GRAPHS_DIR, "*.pkl")) if 'BRCA_' in g]
    
    # Grouping Graphs by Patient ID. In CPTAC multiple WSIs were available for each patients
    # For patients with multiple whole slide images we bagged all WSIs into a single graph
    test_dataset =[]
    patientIDs = set([Path(p).stem.split('-')[0].replace('BRCA_','') for p in graphlist])
    for pid in patientIDs:
        graphsDf = pd.DataFrame(graphlist,columns=['Path'])
        graphs = graphsDf[graphsDf.Path.str.contains(pid)].Path.tolist()
        # Loading each graph
        gBagN = []
        gBagE = []
        gBagC = []
        for i,g in enumerate(graphs):
            G = pickleLoad(g)
            if i>0:
                offset = torch.cat(gBagN[:i]).shape[0]
                gBagE.append(G.edge_index+offset)
                gBagC.append(G.coords+offset)
            else:
                gBagE.append(G.edge_index)
                gBagC.append(G.coords)
            gBagN.append(G.x)
        G.edge_index = torch.cat(gBagE,1)
        G.x = torch.cat(gBagN)
        G.coords = torch.cat(gBagC)
        # G = pickleLoad(g)
        G.pid = pid
        test_dataset.append(G)
        
        ZZ = []
        
        # Generate prediction for each graph using five ensemble models. 
        for f in range(folds):
            print('Prediction ensemble model ',f)
            model = GNN(dim_features=layers[0], dim_target=target_dim,
            layers=layers, dropout=0.1, pooling='mean', conv='EdgeConv', aggr='max')
            net = NetWrapper(model, loss_function=None,
                            device=device,batch_size=batch_size)  # nn.MSELoss()
            model = model.to(device=net.device)
            model.load_state_dict(torch.load(f'{WPATH}/{f}'))
            Z, _, _, Pn = decision_function(model,test_dataset,device=device,returnNumpy=True)
            ZZ.append(list(Z))
        ZZ = np.array(ZZ)
        
        ntopics = 200
        names = ['Patient ID']+[f'T_{i}' for i in range(ntopics)]
        TS = pd.read_csv(f'{DATA_DIR}/cptac_labels.txt',delimiter='\t',
                        names=names,header=None,index_col='Patient ID')
        
        TS.index = [t[1:] for t in TS.index]
        from sklearn.metrics import roc_auc_score
        predictionDf = pd.DataFrame()

        for fold_idx in range(ZZ.shape[0]):
            Pred = pd.DataFrame(ZZ[fold_idx],
                                        columns=[f'P_{topic}_f{fold_idx}' for topic in range(ntopics)])
            Pred.index = Pn
        
            if fold_idx==0:
                predictionDf = Pred
            else:
                predictionDf = predictionDf.join(Pred)

        predictionDf = predictionDf.join(TS).dropna()

        outpath = '/data/PanCancer/HTEX_repo/output/wsi_pred'
        predictionDf.to_csv(f'{outpath}/slideGraph_predictions_cptac.csv')
