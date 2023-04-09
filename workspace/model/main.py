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

if __name__ == '__main__':

    # Saving Node Level Prediction or not
    returnNodeProba = False
    patch_size = (512,512)
    bdir = '/data/PanCancer/HTEX_repo'
    DATA_DIR = f'{bdir}/data'
    TAG = 'x'.join(map(str,patch_size))
    Repr = 'ShuffleNet'
    OUTPUT_DIR = f'{bdir}/Output/{TAG}/{Repr}/' 
            
    GRAPHS_DIR = f'{DATA_DIR}/Graphs/{TAG}/{Repr}/' 

    # Model parameters and hyperparameters
    device = 'cuda:1'
    learning_rate = 0.001
    weight_decay = 0.0001
    epochs = 300 # Total number of epochs
    split_fold =5 # number of folds
    scheduler = None
    batch_size = 8
    layers = [1024,1024,1024]

    # Reading Gene Group Statuses
    nGroups = 200
    names = ['Patient ID']+[i for i in range(nGroups)]
    TS = pd.read_csv(f'{DATA_DIR}/GroupStatuses.txt',names=names,header=None,index_col='Patient ID')

    graphlist = glob(os.path.join(GRAPHS_DIR, "*.pkl"))#[0:200]#0]
    GN = []
    cpu = torch.device('cpu')
    dataset = []
    for graph in tqdm(graphlist):
        patId = graph.split('/')[-1][:15]
        if patId not in TS.index:
            continue
        G = pickleLoad(graph)
        G.to(cpu)
        tStatuses = TS.loc[patId, :].tolist()  # status of all topics
        G.y = toTensor([tStatuses], dtype=torch.float32, requires_grad=False)
        G.pid = patId  # setting up patient id might be used for post processing.
        dataset.append(G)

    print(len(dataset))

    from sklearn.model_selection import KFold
    # Stratified cross validation
    skf = KFold(n_splits=split_fold, shuffle=False)
    Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [
    ], [], [], [], [],  []  # Intialise outputs

    fold = 0
    # Saving Results of Ensemble model
    
    RR = np.full_like(np.zeros((split_fold, TS.shape[1], 2)),np.nan)

    for trvi, test in skf.split(np.arange(0, len(dataset))):
        train, valid = train_test_split(
            trvi, test_size=0.10, shuffle=True)  # ,
    
        train_dataset = [dataset[i] for i in train]
        valid_dataset = [dataset[i] for i in valid]
        v_loader = DataLoader(valid_dataset, shuffle=False)
        test_dataset = [dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)

        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=TS.shape[1],
                    layers=layers, dropout=0.1, pooling='mean', conv='EdgeConv', aggr='max')

        net = NetWrapper(model, loss_function=None,
                         device=device,batch_size=batch_size)  # nn.MSELoss()
        model = model.to(device=net.device)
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        Q, train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc, val_pr, test_pr = net.train(
            train_loader=train_dataset,
            max_epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=20,
            return_best=False,
            log_every=5)
        # Fdata.append((best_model, test_dataset, valid_dataset))
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)
        print("\nfold complete", len(Vacc), train_acc,
              val_acc, tt_acc, val_pr, test_pr)

        print('.....'*20,'Saving Convergence Curve','........'*20)

        path_plot_conv = f'{OUTPUT_DIR}/Converg_Curves/{TAG}/'
        mkdirs(path_plot_conv)
        import matplotlib.pyplot as plt
        ep_loss = np.array(net.history)
        plt.plot(ep_loss)#[:,0]); plt.plot(ep_loss[:,1]); plt.legend(['train','val']);
        plt.savefig(f'{path_plot_conv}/{len(Vacc)}.png')
        plt.close()

        print('.....'*20,'Saving Best model Weights','........'*20)
        weights_path = f'{OUTPUT_DIR}/WEIGHTS/{TAG}'
        mkdirs(weights_path)

        torch.save(Q[0][0].state_dict(), f'{weights_path}/{fold}')

        # Saving node level predictions
        zz, yy, zxn, pn = EnsembleDecisionScoring(
            Q, test_dataset, device=net.device, k=10) # Using 10 ensemble models.

        # saving ensemble results for each fold
        n_classes = zz.shape[-1]
        R = np.full_like(np.zeros((n_classes,2)),np.nan)

        for i in range(n_classes):
            try:
                R[i] = np.array(
                    [calc_roc_auc(yy[:, i], zz[:, i]), calc_pr(yy[:, i], zz[:, i])])
            except:
                print('only one class') 

        df = pd.DataFrame(R, columns=['AUROC', 'AUC-PR'])
        df.index = TS.columns.tolist()
        RR[fold] = R

        res_dir = f'{OUTPUT_DIR}/Results/{TAG}'
        mkdirs(res_dir)
        df.to_csv(f'{res_dir}/{fold}.csv')

        node_pred_dir = f'{OUTPUT_DIR}/nodePredictions/{TAG}/'
        mkdirs(node_pred_dir)

         # saving results of fold prediction
        foldPred = np.hstack((pn[:, np.newaxis], zz, yy))
        foldPredDir = f'{OUTPUT_DIR}/foldPred/{TAG}/'
        mkdirs(foldPredDir)

        columns = ['Patient ID'] +[f'P_{col}' for col in TS.columns]+[f'T_{col}' for col in TS.columns]

        foldDf = pd.DataFrame(foldPred, columns=columns)
        foldDf.set_index('Patient ID', inplace=True)
        foldDf.to_csv(f'{foldPredDir}{fold}.csv')

        if returnNodeProba:
            for i, GG in enumerate(tqdm(test_dataset)):
                G = Data(x=GG.x,edge_index = GG.edge_index,y=GG.y,pid=GG.pid)
                G.to(cpu)
                G.nodeproba = zxn[i][0]
                # adding the target name
                G.class_label = TS.columns.to_numpy()
                G.fc = zxn[i][1]
                ofile = f'{node_pred_dir}/{G.pid}.pkl'
                with open(ofile, 'wb') as f:
                    pickle.dump(G, f)   
        
        # incrementing the fold number
        fold+=1
    # Averaged results of 5 without ensembling
    print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
    print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
    print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
    print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))

    del dataset; 
    import gc; gc.collect()   
    # import pdb; pdb.set_trace()
    RRm = np.nanmean(RR,0)
    RRstd = np.nanstd(RR,0)
    results = pd.DataFrame(np.hstack((RRm, RRstd)))
    results.columns = ['AUROC-mean', 'AUC-PR-mean', 'AUROC-std', 'AUC-PR-std']
    results.index = TS.columns.tolist()
    results.to_csv(f'{OUTPUT_DIR}/Results/{TAG}_restult_stats.csv')
    print('Results written to csv on disk')
    print(results)
