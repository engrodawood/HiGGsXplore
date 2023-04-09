import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# Importing toch libraries 

from wrapper import Helper
from torch.optim import optimizer
from torch.utils.data import Dataset,TensorDataset
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Tanh, LeakyReLU, ELU, SELU, GELU,Sigmoid

device = 'cuda:1'
nworkers=1

def pairwiseLoss(yp,yt):
  loss=torch.tensor(0,dtype=torch.float32).to(device)
  zero = torch.tensor([0],dtype=torch.float32).to(device)
  total_pairs = torch.tensor(0,dtype=torch.float32).to(device)
  for k in range(yp.shape[-1]):
    y_pred = yp[:,k]
    y = yt[:,k]
    # filtering examples with unknown target
    vidx = ~torch.isnan(y)
    y = y[vidx]
    y_pred = y_pred[vidx]
    dY = (y.unsqueeze(1) - y)
    dZ = (y_pred.unsqueeze(1) - y_pred)[dY>0]
    dY = dY[dY>0]
    if len(dY)>0:
      loss+= torch.nanmean(torch.max(zero,1.0-dY*dZ))
      total_pairs+=1
  return loss/total_pairs

class MLP(nn.Module):
  '''
  A simple Perceptor than takes topic predictions as input and predict
  the target clinical variable of interest as output. 
  '''
  def __init__(self,feats=None,targets=None):
      super(MLP, self).__init__()
      self.layers = nn.Sequential(
          nn.Linear(len(feats), len(targets))
      )
      
  def forward(self, x):
      x = self.layers(x)
      return x


# Constant variables and path to data and predictions
batch_size = 256
patch_size = (512,512)
bdir = '/data/PanCancer/HTEX_repo'
DATA_DIR = f'{bdir}/data'
TAG = 'x'.join(map(str,patch_size))
Repr = 'ShuffleNet'
OUTPUT_DIR = f'{bdir}/OUTPUT/{TAG}/{Repr}/' 

predictions_dir = f'{OUTPUT_DIR}/foldPred/'

def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_predictions(test_fold=0):

    '''
    Function will return dataframe with both true and predicted topic status for the whole
    set if you want to reduce the set size set Filter to True
    '''
    # Loading model prediction results for different fold
    predDf = pd.DataFrame()
    trainPidx,testPidx = [],[]
    print('Loading data of Fold ',test_fold)
    for fold in range(5):
        tmpdf = pd.read_csv('{:s}{:d}.csv'.format(predictions_dir,fold)).set_index('Patient ID')
        predDf = pd.concat([predDf,tmpdf])
        if fold == test_fold:
          testPidx.extend(tmpdf.index.tolist())
        else:
            trainPidx.extend(tmpdf.index.tolist())
    return predDf,trainPidx,testPidx


# Loading ER,PR, HER2 status
tissue = 'BRCA'
BDIR = f'/data/PanCancer/Topic_Modeling/{tissue}'
RECEPTOR_STATUS_DATA_DIR = f'{BDIR}/data/receptor_status'
clin_file =  'nationwidechildrens.org_clinical_patient_brca (4).txt'
cols_to_filter = ['er_status_by_ihc', 'pr_status_by_ihc','her2_status_by_ihc']
RecStatus = pd.read_csv(f'{RECEPTOR_STATUS_DATA_DIR}/{clin_file}',delimiter='\t')
RecStatus.rename(columns = {'bcr_patient_barcode':'Patient ID'},inplace=True)
RecStatus.set_index('Patient ID',inplace=True)
RecStatus = RecStatus.loc[:,cols_to_filter]

# Converting string value to numeric
statusClean = {
    'Positive':1,'Negative':0,'Indeterminate':np.nan,'[Not Evaluated]':np.nan,
    'Equivocal':np.nan, '[Not Available]':np.nan
}
RecStatus.replace(statusClean,inplace=True)
RecStatus = RecStatus.iloc[2:,:]

# Reading gen point mutation status
MUT_STATUS_DATA_DIR = f'{BDIR}/data/point_mutation_status/'
mut_file = 'sample_matrix_mutation.txt'
MUTStatus = pd.read_csv(f'{MUT_STATUS_DATA_DIR}/{mut_file}',delimiter='\t')
mutKey = 'studyID:sampleId'
MUTStatus.rename(columns = {'bcr_patient_barcode':'Patient ID'},inplace=True)
MUTStatus['Patient ID'] = [idx.split(':')[1][:12] for idx in MUTStatus.iloc[:,0].tolist()]
MUTStatus.set_index('Patient ID',inplace=True)
MUTStatus.drop(columns=[mutKey],inplace=True)
MUTStatus.columns = [f'{c} (MUT)' for c in MUTStatus]

# Reading Genes Copy number alteration status
CNA_STATUS_DATA_DIR = f'{BDIR}/data/cna_status'
cna_file = 'sample_matrix_alter.txt'
CNAStatus = pd.read_csv(f'{CNA_STATUS_DATA_DIR}/{cna_file}',delimiter='\t')
mutKey = 'studyID:sampleId'
CNAStatus.rename(columns = {'bcr_patient_barcode':'Patient ID'},inplace=True)
CNAStatus['Patient ID'] = [idx.split(':')[1][:12] for idx in CNAStatus.iloc[:,0].tolist()]
CNAStatus.set_index('Patient ID',inplace=True)
CNAStatus.drop(columns=[mutKey,'Altered'],inplace=True)
CNAStatus.columns = [f'{c} (CNV)' for c in CNAStatus]

# Reading Pathway status
PTHY_STATUS_DATA_DIR = f'{BDIR}/data/curated_pathways'
pathway_file = 'mmc4.xlsx'
PTHYStatus = pd.read_excel(f'{PTHY_STATUS_DATA_DIR}/{pathway_file}','Pathway level')
PTHYStatus['Patient ID'] = [p[:12] for p in PTHYStatus.SAMPLE_BARCODE]
PTHYStatus.set_index('Patient ID',inplace=True)
PTHYStatus.drop(columns=['SAMPLE_BARCODE'],inplace=True)


# Reading Immune Subtypes
DATA_DIR = f'{BDIR}/data/immune_subtypes'
clin_file =  'TCGA-BRCA-DX_CLINI.xlsx'
CD = pd.read_excel(f'{DATA_DIR}/{clin_file}')
CD.rename(columns = {'PATIENT':'Patient ID'},inplace=True)
CD.set_index('Patient ID',inplace=True)
# One-hot encoding of target variables
target = 'ImmuneSubtype'
ImStype = CD.loc[:,target].dropna()
imTypes = sorted(list(set(ImStype.tolist())))
ImmuneDf = pd.DataFrame(np.zeros((ImStype.shape[0],len(imTypes))),
                          index = ImStype.index,
                          columns=imTypes
                          )
# One hot encoding of TCGASubtypes
target = 'TCGASubtype'
for imType in imTypes:
  ImmuneDf.loc[ImStype==imType,imType] = 1.0
PAMStype = CD.loc[:,target].dropna()
PAMTypes = sorted(list(set(PAMStype.tolist())))
PAMDf = pd.DataFrame(np.zeros((PAMStype.shape[0],len(PAMTypes))),
                          index = PAMStype.index,
                          columns=PAMTypes
                          )
for pamType in PAMTypes:
  PAMDf.loc[PAMStype==pamType,pamType] = 1.0

# Using Predicted Gene Groups status as features. 
feats = [f'P_{t}' for t in range(200)]

# Dictionary Mapping of target variables

target_dict = {}
target['Receptor Status'] = RecStatus.columns.tolist()
target['PAM50 Subtypes'] = PAMDf.columns.tolist()
target['Immune Subtypes'] = ImmuneDf.columns.tolist()
target['MUT Status'] = MUTStatus.columns.tolist()
target['CNA Status'] = CNAStatus.columns.tolist()
target['Pathways Status'] = PTHYStatus.columns.tolist()


import pdb; pdb.set_trace()
targets = list(PAMDf.columns)#list(PAMDf.columns)#+list(MUTStatus.columns)+list(RecStatus)+list(PTHYStatus)
n_folds = 5
resDf = pd.DataFrame(np.zeros((n_folds,len(targets)))*np.nan,
                      index=np.arange(0,n_folds),
                      columns=targets
                    )
valAucDf = resDf.copy()
testAucDf = resDf.copy()
Wnpy = np.zeros((n_folds,len(feats),len(targets)))

TAG_RUN = f'PAM50Types'#_DY_gt_0' # Originally it was 256
outPath = '/data/PanCancer/TopicPrediction/BRCA/Topic_SlideGraph/downstream_analysis'
YsPred = []
YsTrue = []
YsPids = []
for fold_idx in range(n_folds):
  predDf,trainPidx,testPidx= load_predictions(test_fold=fold_idx)
  #dataDf = predDf.join(RecStatus)#.join(MUTStatus).join(CNAStatus)
  #dataDf = predDf.join(RecStatus)#.dropna()
  # dataDf = predDf.join(CNAStatus)
  #dataDf = predDf.join(MUTStatus)
  # dataDf = predDf.join(PTHYStatus)
  #dataDf = predDf.join(ImmuneDf)
  dataDf = predDf.join(PAMDf)

  testDf = dataDf.loc[testPidx,:]#.dropna()
  trainDf = dataDf.loc[trainPidx,:]#.dropna()
  X,Y = trainDf[feats].to_numpy().astype('float32'),trainDf[targets].to_numpy().astype('float32')
  Xts,Yts = testDf[feats].to_numpy().astype('float32'),testDf[targets].to_numpy().astype('float32')
  # Using 10% data from train for validation
  Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=0.10,shuffle=True)
  # import pdb; pdb.set_trace()

  # train validation, and test loader
  train_data = TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ytr))
  val_data = TensorDataset(torch.from_numpy(Xv),torch.from_numpy(Yv))
  test_data = TensorDataset(torch.from_numpy(Xts),torch.from_numpy(Yts))
  
  train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True,pin_memory=False,num_workers=nworkers)
  validation_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size,pin_memory=False,num_workers=nworkers,shuffle=False)
  model = MLP(feats=feats,targets=targets)
  model.to(device)
  criterion = pairwiseLoss#torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.0001)#momentum=0.9,nesterov=True)#1)#.001)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
  wrapper = Helper(model,criterion,optimizer,tr_loader=train_loader,
  early_stopping_thresold=20,
  # checkpoint_path=f'{outPath}/{TAG_RUN}_chkp',
  # best_path = f'{outPath}/{TAG_RUN}_best',
  save_best=False,
  vl_loader=validation_loader,device=device,lr_scheduler=lr_scheduler,early_stopping=True)

  Tloss, Vloss = [],[]
  # Model Training
  for epoch in range(1000):
    if wrapper.training==0: break
    tl = Helper.train(wrapper)
    vl = Helper.validation(wrapper,epoch)
    Tloss.append(tl); Vloss.append(vl)
    print('Epoch: ',epoch, ' Train Loss ',tl, ' Vlaidation Loss ', vl )

  # Iterating through weights learned by the model
  for idx,name in enumerate(targets):
    # import pdb; pdb.set_trace()
    Wnpy[fold_idx,:,idx] = model.layers[0].weight[idx].detach().cpu().numpy()
  # Generating Inference on the Test Set
  
  test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,pin_memory=False,num_workers=nworkers,shuffle=False)
  Yvp,Yv = Helper.evaluate(wrapper,validation_loader)
  Ytp,Yts = Helper.evaluate(wrapper,test_loader)

  YsPred.extend(list(Ytp))
  YsTrue.extend(list(Yts))
  YsPids.extend(testDf.index.tolist())
  print(np.shape(YsPred),np.shape(YsTrue),np.shape(YsPids))

  # for a given fold iterate across each target

  for tidx,targ in enumerate(targets):
    if len(targets)<=1:
            # Select only one probabity for auroc
            Yvp = Yvp[:,1]
            Ytp = Ytp[:,1]
            Yvp = Yvp[:,np.newaxis]
            Ytp = Ytp[:,np.newaxis]

    #Computing Featurewise AUROC
    for idx,k in enumerate(targets):
        try:
          # Filtering nan-value
          valAucDf.loc[fold_idx,k] = roc_auc_score(Yv[~np.isnan(Yv[:,idx]),idx],
                                                    Yvp[~np.isnan(Yv[:,idx]),idx])
          testAucDf.loc[fold_idx,k] = roc_auc_score(Yts[~np.isnan(Yts[:,idx]),idx],
                                                    Ytp[~np.isnan(Yts[:,idx]),idx])
        except:
          print('single label')
DD = pd.DataFrame(Wnpy.mean(0),index=feats,columns=targets)
DD.to_csv(f'{outPath}/WEIGHTS/{TAG_RUN}.csv')
# import seaborn as sns
# sns.clustermap(DD)
# plt.show()
# import pdb; pdb.set_trace()
# Saving fold level AUROC Results
valAucDf.to_csv(f'{outPath}/FoldRes/{TAG_RUN}_val.csv')
testAucDf.to_csv(f'{outPath}/FoldRes/{TAG_RUN}_test.csv')

print('validation Results')
valResults = pd.DataFrame()
valResults['mean (AUROC)'] = valAucDf.mean()
valResults['std (AUROC)'] = valAucDf.std()
valResults.index = valAucDf.columns
print(valResults)
fodPredPath = f'{outPath}'
valResults.to_csv(f'{outPath}/AUROC/{TAG_RUN}_val.csv')
testResults = pd.DataFrame()
testResults['mean (AUROC)'] = testAucDf.mean()
testResults['std (AUROC)'] = testAucDf.std()
testResults.index = testAucDf.columns
print(testResults)
testResults.to_csv(f'{outPath}/AUROC/{TAG_RUN}_test.csv')

PDf = pd.DataFrame(
                  data = np.hstack((YsTrue,YsPred)),
                  columns = [f'T_{lbl}' for lbl in PAMDf.columns]+[f'P_{lbl}' for lbl in PAMDf.columns],
                  index=YsPids
                   )
# Saving test set prediction
PDf.to_csv(f'{outPath}/Pred/{TAG_RUN}_test.csv')




