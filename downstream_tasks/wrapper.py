import os
import numpy as np 
import torch
import shutil

class Helper:
  def __init__(self,model,criterion, optimizer, tr_loader=None, vl_loader=None, device= torch.device('cuda:1'),
      checkpoint_path=None, best_path=None, save_best=False,early_stopping=False,
      early_stopping_thresold=10,lr_scheduler=None):
      self.model = model
      self.criterion = criterion
      self.checkpoint_path = checkpoint_path
      self.best_model_path = best_path
      self.save_best = save_best
      self.optim = optimizer
      self.device = device
      self.tr_loader = tr_loader
      self.vloader = vl_loader
      self.early_stopping=early_stopping
      self.lr_scheduler = lr_scheduler 
      self.early_stopping_thresold = early_stopping_thresold
      self.vloss_monitor = torch.inf
      self.training = early_stopping_thresold
      

  def save_ckp(self,epoch):
    """
    state: checkpoint state dictionary to be saved
    """
    f_path = f'{self.checkpoint_path}/checkpoint_{epoch}.pt'
    if not os.path.isdir(self.checkpoint_path):
        os.makedirs(self.checkpoint_path)
    # save checkpoint data to the path given, checkpoint_path
    torch.save(self.model.state_dict(), f_path)
    # if it is a best model, min validation loss
    if self.save_best:
        if not os.path.isdir(self.best_model_path):
            os.makedirs(self.best_model_path)
        best_fpath = f'{self.best_model_path}/best_model.pt'
        shutil.copyfile(f_path, best_fpath)
  
  def load_best(self):
      """
      Load check point from a given path accept .pt file
      """
      checkpoint = torch.load(f'{self.best_model_path}/best_model.pt')
      return self.model.load_state_dict(checkpoint)

  def train(self):
    L=[]
    self.model.train()
    for (data, target) in (self.tr_loader):
      data, target = data.to(self.device), target.to(self.device)
      self.optim.zero_grad()
      #import pdb; pdb.set_trace()
      output = self.model(data)
      loss = self.criterion(output, target)
      L.append(loss.item())
      loss.backward()
      self.optim.step()
    return np.round(np.nanmean(L),6)

  def validation(self,epoch):
    L=[]
    self.model.eval()
    for (data, target) in (self.vloader):
      data, target = data.to(self.device), target.to(self.device)
      # import pdb; pdb.set_trace()
      output = self.model(data)
      loss = self.criterion(output, target)
      L.append(loss.item())
    l = np.round(np.nanmean(L),6)
    if self.lr_scheduler:
      self.lr_scheduler.step(l)
    if self.early_stopping:
      if l < self.vloss_monitor:
        # self.save_ckp(epoch)
        self.vloss_monitor=l
        self.training = self.early_stopping_thresold
      else:
        self.training-=1
    return l

  def evaluate(self, test_loader,infer_best=False):
      """
      Input:
      model: model object
      test_loader: Data loader object
      Output: 
      return scores and target label
      """
      if infer_best:
        self.load_best()
      scores=[]
      targets=[]
      self.model.eval()
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(self.device), target.to(self.device)
              Z = self.model(data)
              scores.extend(Z.detach().cpu().numpy())
              targets.extend(target.detach().cpu().numpy())
          return np.array(scores),np.array(targets)