import torch as th
import numpy as np
import random
import h5py
import os
import yaml
import tqdm
from dr_modules import utils
import time

from collections import namedtuple
def re_namedtuple(pd):
  if isinstance(pd,dict):
    for key in pd:
      if isinstance(pd[key],dict):
        pd[key] = re_namedtuple(pd[key])
    pod = namedtuple('cfg',pd.keys())
    bobo = pod(**pd)
  return bobo
"""
point 
  phi theta e_s e_c time
  self.xmax = [1100, 3800, 15, 25, 300] 
reco3d 
  phi theta time e_s e_c cid
  self.xmax = [1100, 3800, 15, 25, 300] 
sim3d 
  x y z en time pdg cid
  DRcalo3dHits.cellID 25 0
  DRcalo3dHits.energy 14.007626 0.0
  DRcalo3dHits.ix 54 0
  DRcalo3dHits.iy 54 0
  DRcalo3dHits.noPhi 282 0
  DRcalo3dHits.noTheta 91 -92
  DRcalo3dHits.phi_idx 7849 -7688
  DRcalo3dHits.theta_idx 4153 -4153
  DRcalo3dHits.time 84.9 0.0
  DRcalo3dHits.type 1 0
  E_dep 109.7924575805664 0.0
  GenParticles.PDG 211 11
  GenParticles.momentum.p 109.96124267578125 10.000219345092773
  GenParticles.momentum.phi 0.22193138301372528 -0.22198565785390784
  GenParticles.momentum.theta 1.7149051427841187 0.20454877614974976
  Leakages.PDG 1000010030 -2212
  Leakages.momentum.p 60.201799783990815 0.0
  Leakages.momentum.phi 1.5707930326461792 -1.5707961320877075
  Leakages.momentum.theta 3.1409947872161865 0.0
  SimCalorimeterHits.cellID 2318304 0
  SimCalorimeterHits.energy 93.972984 0.0
  SimCalorimeterHits.position.x 3800.0266 -3799.7925
  SimCalorimeterHits.position.y 3799.968 -3799.968
  SimCalorimeterHits.position.z 4544.4907 -4544.4907
"""
def apply_jitter(batch_data,is_point=True): 
    """ Randomly jitter points. jittering is per point. 
    """ 
    sigma=batch_data.shape[-1]*np.mean(batch_data)/10
    clip=5*sigma
    #assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(*batch_data.shape), -1 * clip, clip)
    if(is_point):
        jittered_data[:,:,0]=np.round(jittered_data[:,:,0])
        jittered_data[:,:,1]=np.round(jittered_data[:,:,1])
        jittered_data[:,:,4]=np.round(jittered_data[:,:,4])
    jittered_data += batch_data
    return jittered_data

class h5pyData(th.utils.data.Dataset):
  def __init__(self,cand,is_train=False,is_val=False,path=None,args=None,balance=None,config=None):
    if(path is None):
      path="/pad/yulee/torch/h5s"
    if(path=="/pad/yulee/torch/h5s"):
        reco=True
    else:
        reco=False
    print("path",path)
    #time.sleep(10)
    self.siglabel=[]
    self.cand=cand
    self.target=config.target
    self.target_scale=config.target_scale
    self.h5path=[]
    self.key=None
    self.ch=None
    if('theta' in self.target):
        self.flip=True
        self.THETA_SCALE=self.target_scale[self.target.index('theta')]
    else:
        self.flip=False
    #if('P' in self.target or 'dep' in self.target):
    #    self.noise=True
    #else:
    #    self.noise=False
    self.noise=False
    self.dform="point"
    self.dform="reco3d"
    self.num_point=config.num_point
    #self.pool=config.pool
    self.channel=[
                'DRcalo3dHits.energy',
                'DRcalo3dHits.type',
                'DRcalo3dHits.time',
                'DRcalo3dHits.phi_idx',
                'DRcalo3dHits.theta_idx',
                ]
    self.channelmax={
                'DRcalo3dHits.energy':1,
                'DRcalo3dHits.type':1,
                'DRcalo3dHits.time':1,
                'DRcalo3dHits.phi_idx':1,
                'DRcalo3dHits.theta_idx':1,
                }
    self.num_ch=len(self.channel)
    load=[]
    for i in range(len(self.cand)):
        name=self.cand[i]
        print(f'{path}/{name}.h5py')
        if(os.path.isfile(f'{path}/{name}.h5py')):
            load.append(h5py.File(f'{path}/{name}.h5py', 'r'))
            self.h5path.append(f'{path}/{name}.h5py')
        else:
            load.append(h5py.File(f'/store/ml/dual-readout/h5s/{name}.h5py', 'r'))
            self.h5path.append(f'/store/ml/dual-readout/h5s/{name}.h5py')
        label=[0. for j in range(len(self.cand))]
        label[i]=1.
        self.siglabel.append(label)
    if('E_dep' in load[0].keys()):
        entries=[len(load[i]['E_dep']) for i in range(len(load))]
    elif('gen_pdg' in load[0].keys()):
        entries=[len(load[i]['gen_pdg']) for i in range(len(load))]
    else:
        entries=[len(load[i]['prt']) for i in range(len(load))]
    if(balance is None):
      self.balance=min(entries)
    else:
      self.balance=min([balance]+entries)
    self.is_train=is_train
    self.is_val=is_val
    kf=4
    self.begin=[]
    self.end=[]
    for i in range(kf):
        cbegin=[]
        cend=[]
        for j in range(len(self.cand)):
            bbin=self.balance/kf
            start=self.balance/kf*i
            #bbin=entries[j]/kf
            #start=entries[j]/kf*i
            if(is_train):
                cbegin.append(int(start))
                cend.append(int(start+bbin*0.7*0.7))
            elif(is_val):
                cbegin.append(int(start+bbin*0.7*0.7))
                cend.append(int(start+bbin*0.7))
            else:
                cbegin.append(int(start+bbin*0.7))
                cend.append(int(start+bbin))
        self.begin.append(cbegin)
        self.end.append(cend)
        #self.target=['phi','theta','P','dep']
    self.set_num=[]
    self.set_idx=[]
    thetas=[]
    self.Xdata=[]
    self.Y1data=[]
    self.Y2data=[]
    self.origin=[]
    for k in tqdm.tqdm(range(kf),desc="data loader"):
        for j in tqdm.tqdm(range(len(self.cand)),leave=False):
            begin=self.begin[k][j]
            end=self.end[k][j]
            if(not os.getenv("TQDM_DISABLE") is None):
                print(k,self.cand[j])
            Xcand=[]
            for c in self.channel:
                Xcand.append(load[j][c][begin:end,:self.num_point])
            Xcand=np.array(Xcand).transpose((1,2,0))
            Y1=[]
            Y2cand=[]
            for t in self.target:
                Y2cand.append(load[j][t][begin:end].reshape(-1))
            Y2cand=np.array(Y2cand).transpose((1,0))
            for i in tqdm.tqdm(range(self.end[k][j]-self.begin[k][j]),leave=False):
                self.Xdata.append(Xcand[i])
                self.origin.append([j,i+begin])
                self.Y1data.append(self.siglabel[j])
                self.Y2data.append(Y2cand[i]*self.target_scale)
    for j in range(len(self.cand)):
        load[j].close()
    self.Xdata=np.array(self.Xdata)
    #self.xmax=[np.max(abs(self.Xdata[:,0,i])) for i in range(self.num_ch)]
    self.xmax = [self.channelmax[c] for c in self.channel]
    #self.xmax = [1100, 3800, 15, 25, 300] 
    self.normalize()
  def normalize(self,):
    for i in range(self.num_ch):
        self.Xdata[:,:,i]=self.Xdata[:,:,i]/self.xmax[i]
    """
    for j in range(len(self.cand)):
        thetas.append(load[j]['genPptp'][self.begin:self.end])
    for i in range(self.begin,self.end):
        for j in range(len(self.cand)):
            if(thetas[j][i-self.begin][1]<0.25):continue
            self.set_idx.append(i)
            self.set_num.append(j)
    for i in range(len(self.cand)):
        load[i].close()
    """

  def __getitem__(self, index):
    Xbuf=self.Xdata[index]
    Y2buf=self.Y2data[index]
    #if(self.num_ch>2 and self.is_train and self.flip):
    #    if(random.random()>0.5):#theta flip
    #        Xbuf[:,1]=-Xbuf[:,1]
    #        Y2buf[1]=np.pi-Y2buf[1]+0.006*self.THETA_SCALE
    #if(self.num_ch>3 and self.is_train and self.noise):
    #    Xbuf[:,2]=Xbuf[:,2]*(1+np.random.randn(self.num_point)*0.0001)
    #    Xbuf[:,3]=Xbuf[:,3]*(1+np.random.randn(self.num_point)*0.0001)
    if(self.is_train or self.is_val):
      return th.tensor(Xbuf).float(),th.tensor(self.Y1data[index]).float(),th.tensor(Y2buf).float()
    else:
      return th.tensor(Xbuf).float(),th.tensor(self.Y1data[index]).float(),th.tensor(Y2buf).float(),th.tensor(self.origin[index]).float()

  def __len__(self):
    return len(self.Y1data)
    #return len(self.set_idx)
#normalize()
#standardscale()

class RandomFlip(object):
    def __init__(self,axis=[0,1]):
        self.axis=axis
    def __call__(self,x):
        b = x.size()[0]
        for i in range(b):
            if random.random()<0.95:
                for ax in self.axis:
                    if random.random() < 0.5:
                        coords[i,:,ax] = -coords[i,:,ax]
        return coords
        


if __name__== '__main__':
    config="""
batch_size: 64
num_point: 2048
input_channel: 3
cls_dim: 2
emb_dim: 32
depth: 2
num_head: 4
group_size: 16
num_group: 16
trans_dim: 32
encoder_dims: 32
encoder_feature: 32
fine_dim: 128
rms_norm: False
drop_path: 0.2
drop_out: 0.1
"""
    cand=["e-_line","pi+_line"]
    balance=1000
    cfg=utils.re_namedtuple(yaml.safe_load(config))
    batch_size=cfg.batch_size
    train_dataset=h5pyData(cand,is_train=True,balance=balance,config=cfg)
    train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=len(cand),shuffle=True)
    print(len(train_loader))
    X,Y1,Y2=next(iter(train_loader))
    print(X.shape,Y1.shape,Y2.shape)
    print(Y1,Y2)
