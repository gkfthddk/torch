#!/pad/yulee/micromamba/envs/torch212/bin/python3
import os
import argparse

cands=['e-','gamma','pi0','pi+','kaon+','kaon0L','neutron','proton']
args=None
if __name__== '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--balance",type=int,default=None,help='eval epoch')
    parser.add_argument("--epoch",type=int,default=200,help='eval epoch')
    parser.add_argument("-em","--epoch_min",type=int,default=20,help='minimum epoch threshold')
    parser.add_argument("--num_point",type=int,default=1000,help='num_point')
    parser.add_argument("--pool",type=int,default=1,help='pool')
    parser.add_argument("--depth",type=int,default=8,help='depth')
    parser.add_argument("--encoder_feat",type=int,default=128,help='encoder_dim')
    parser.add_argument("--encoder_dim",type=int,default=128,help='encoder_dim')
    parser.add_argument("--emb_dim",type=int,default=64,help='emb_dim')
    parser.add_argument("--fine_dim",type=int,default=512,help='encoder_dim')
    parser.add_argument("--last_dim",type=int,default=64,help='encoder_dim')
    parser.add_argument("--in_channel",type=int,default=5,help='in channel')
    parser.add_argument("--group_size",type=int,default=32,help='emb_dim')
    parser.add_argument("--num_group",type=int,default=64,help='emb_dim')
    parser.add_argument("--gpu",type=str,default='2',help='gpu')
    parser.add_argument("--model",type=str,default='MLP',help='model type MLP Mamba Conv')
    parser.add_argument("--name",type=str,default='mamba_id',help='model name')
    parser.add_argument("--eval", action='store_true', help="set eval mode")
    parser.add_argument("--condor", action='store_true', help="check is condor")
    parser.add_argument("--cand",type=str,default=",".join([f'{c}' for c in cands]),help='candidate fileset')
    parser.add_argument(
        "--target",type=str,
        default="E_dep,GenParticles.momentum.phi,GenParticles.momentum.theta,Leak_sum",
        #default="E_dep,GenParticles.momentum.p,GenParticles.momentum.phi,GenParticles.momentum.theta",
        help='regression target variables')
    parser.add_argument("--target_scale",type=str,default=None,help='target variable scale')
    parser.add_argument("--path",type=str,default=None,help='cand')
    args=parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if(not args.condor):
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
if(args.condor):
    if(not os.path.isdir("save")):
        os.makedirs("save")
    os.environ["TQDM_DISABLE"]="1"
#else:os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
print(" ".join(sys.argv[1:]))
import json
import yaml
import torch as th
assert th.cuda.is_available()
from torchinfo import summary
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import MeanSquaredError
import numpy as np
import gc
import datetime
import h5py
import tqdm

from nh5 import *
from pm import *

th.autograd.set_detect_anomaly(True)

class metric:
    def __init__(self,name):
        self.name=name
        self.metric1=BinaryAUROC(device='cuda')
        self.metric2=MeanSquaredError(device='cuda')
        self.n=0
        self.loss=0
        self.weight=None
    def reset(self,):
        self.n=0
        self.loss=0
        self.metric1.reset()
        self.metric2.reset()
        self.weight=None
    def update(self,output,target,loss_item,weight=None):
        self.metric1.update(output[0][:].ravel(),target[0][:].ravel())
        for i in range(1,len(output)):
            if(output[i] is None): continue
            self.metric2.update(output[i].ravel(),target[1][:,i-1])
        self.n += len(output[0])
        self.loss+=loss_item*len(output[0])
        self.weight=weight
    def getloss(self,):
        return self.loss/self.n
    def getdesc(self,):
        if self.weight is not None:
            weight_text=",".join([f"{self.weight[i]:.2f}" for i in range(len(self.weight))])
            return f"{self.name} weight=({weight_text}),loss=({self.loss/self.n:.3f}), auc {self.metric1.compute().item():.3f}, mse {self.metric2.compute().item():.3f}"
        else:
            return f"{self.name} loss=({self.loss/self.n:.3f}), auc {self.metric1.compute().item():.3f}, mse {self.metric2.compute().item():.3f}"

        
class model:
    def __init__(self,name,config,condor=False):
        self.config=config
        self.name=name
        self.condor=condor
    def setnet(self,net):
        self.net=net
        self.optim = th.optim.AdamW(self.net.parameters(), lr=0.0003)

    def train(self,train_loader,val_loader=None,num_epoch=100):
        train_metric=metric("train")
        val_metric=metric("val")
        best_loss=2.
        early_stop=0
        check=0
        for epoch in range(num_epoch):
            train_metric.reset()
            th.set_grad_enabled(True)
            if(not self.condor):
                bar = tqdm.tqdm(total=len(train_loader),nrows=2,leave=val_loader is None)
            self.net.train()
            for X,Y1,Y2 in train_loader:
                X = X.to("cuda", non_blocking=True)
                Y1 = Y1.to("cuda", non_blocking=True)
                Y2 = Y2.to("cuda", non_blocking=True)
                if(check==0):
                    print(Y2[:5])
                    check=1
                output = self.net(X)

                self.optim.zero_grad()
                #loss.backward(retain_graph=True)

                try:
                    #loss.backward()
                    if(epoch<10):loss=self.net.get_loss(output,[Y1,Y2])
                    else:loss=self.net.gradnorm_loss(output,[Y1,Y2])
                except:
                    print("X",th.any(th.isnan(X)))
                    print("Y1 Y2",th.any(th.isnan(Y1)),th.any(th.isnan(Y2)))
                    print("out",th.any(th.isnan(output[0])),th.any(th.isnan(th.Tensor(output[1]))))
                    if(epoch<10):loss=self.net.get_loss(output,[Y1,Y2])
                    else:loss=self.net.gradnorm_loss(output,[Y1,Y2])
                    print("loss",th.any(th.isnan(loss)))
                    break
                train_metric.update(output,[Y1,Y2],loss.item(),self.net.weight)

                self.optim.step()

                if(not self.condor):
                    bar.set_description(f"epoch{epoch:4d} : {train_metric.getdesc()} ")
                    bar.update()
            if(self.condor):
                print(f"epoch{epoch:4d} : {train_metric.getdesc()} ")
            gc.collect()
            if(not val_loader is None):
                val_metric.reset()
                if(not self.condor):
                    bar = tqdm.tqdm(total=len(val_loader),nrows=2)
                self.net.eval()
                th.set_grad_enabled(False)
                for X,Y1,Y2 in val_loader:
                    X = X.to("cuda", non_blocking=True)
                    Y1 = Y1.to("cuda", non_blocking=True)
                    Y2 = Y2.to("cuda", non_blocking=True)
                    output = self.net(X)
                    if(check==1):
                        print(Y2[:5])
                        check=2

                    loss = self.net.get_loss(output,[Y1,Y2])
                    val_metric.update(output,[Y1,Y2],loss.item())

                    if(not self.condor):
                        bar.set_description(f"epoch{epoch:4d} : {train_metric.getdesc()} {val_metric.getdesc()} ")
                        bar.update()
                if(self.condor):
                    print(f"epoch{epoch:4d} : {train_metric.getdesc()} {val_metric.getdesc()} ")
                if(val_metric.getloss()<best_loss or epoch==30):
                  early_stop=0
                  best_loss=val_metric.getloss()
                  self.save(epoch,best_loss)
                if(epoch>args.epoch_min and early_stop>7):break
                else:early_stop+=1
    def save(self,epoch,best_loss):
        if(not os.path.isdir("save")):
          os.makedirs("save")
        th.save({'net':self.net.state_dict(),'optimizer':self.optim.state_dict(),'best_loss':best_loss,'epoch':epoch,'config':self.config},f'save/ckpt-{self.name}-best.pth')
    def loadconfig(self,):
        tl=th.load(f'save/ckpt-{self.name}-best.pth')
        self.config=(tl['config'])
    def loadnet(self):
        tl=th.load(f'save/ckpt-{self.name}-best.pth')
        self.net.load_state_dict(tl['net'])
        print("best loss",tl['best_loss'])
    def test(self,test_loader):                
        th.set_grad_enabled(False)
        self.net.eval()
        hf=h5py.File(f'save/{self.name}_best.h5py', 'w')
        #test_X=hf.create_dataset("test_X",shape=(len(test_dataset),X.shape[0],X.shape[1]),dtype='float32')
        X,Y1,Y2,origin=test_loader.dataset[0]
        test_Y1=hf.create_dataset("test_Y1",shape=(len(test_dataset),Y1.shape[0]),dtype='float32')
        test_Y2=hf.create_dataset("test_Y2",shape=(len(test_dataset),Y2.shape[0]),dtype='float32')
        test_O=hf.create_dataset("test_O",shape=(len(test_dataset),origin.shape[0]),dtype='int32')
        test_p1=hf.create_dataset("test_p1",shape=(len(test_dataset),Y1.shape[0]),dtype='float32')
        test_p2=hf.create_dataset("test_p2",shape=(len(test_dataset),Y2.shape[0]),dtype='float32')
        if(not self.condor):
            bar = tqdm.tqdm(total=len(test_loader),nrows=2)
        test_metric=metric("test")
        tloss=0
        n_t=0
        for X,Y1,Y2,origin in test_loader:
            #test_X[n:n+len(X)]=X
            test_O[n_t:n_t+len(X)]=origin
            test_Y1[n_t:n_t+len(X)]=Y1
            test_Y2[n_t:n_t+len(X)]=Y2[:]
            X = X.to("cuda")
            Y1 = Y1.to("cuda")
            Y2 = Y2.to("cuda")
            #Y2 = Y2[:,2:3].to("cuda")
            output = self.net(X)
            loss = self.net.get_loss(output,[Y1,Y2])
            test_metric.update(output,[Y1,Y2],loss.item())
            test_p1[n_t:n_t+len(X)]=th.nn.functional.softmax(output[0],dim=1).cpu().numpy()
            test_p2[n_t:n_t+len(X)]=output[1].cpu().numpy()
            tloss+=loss.item()*len(output[0])
            n_t+=len(output[0])
            if(not self.condor):
                bar.set_description(f"{test_metric.getdesc()}")
                bar.update()
        if(self.condor):
            print(f"{test_metric.getdesc()}")

        cfg=re_namedtuple(yaml.safe_load(self.config))
        hf.attrs['cand']=",".join(cfg.cand)
        hf.attrs['target']=",".join(cfg.target)
        hf.attrs['target_scale']=",".join([str(t) for t in cfg.target_scale])
        hf.close()
        print(f'save/{self.name}_best.h5py')



net=None
if __name__== '__main__':
    SCALE={
        "E_dep":0.1,
        'GenParticles.momentum.p':0.1,
        'GenParticles.momentum.phi':1.,
        'GenParticles.momentum.theta':1.,
        'Leak_sum':1.,
    }
    start_time=datetime.datetime.now()
    print(start_time)
    name=args.name
    if("{c}" in args.cand):
        cand = [args.cand.replace("{c}",c) for c in cands]
    else:
        cand = args.cand.split(",")
    in_channel = args.in_channel
    #target = ['phi', 'theta', 'P']
    target = args.target.split(",")
    
    if(args.target_scale is None):
        target_scale=[SCALE[t] for t in target]
    else:
        target_scale = [float(t) for t in args.target_scale.split(",")]
    config=f"""
NAME: {name}
cand: {cand}
batch_size: 64
target: {target}
target_scale: {target_scale}
num_point: {args.num_point}
pool: {args.pool}
model:
  NAME: {args.model}
  in_channel: {in_channel}
  cls_dim: {len(cand)}
  reg_dim: {len(target)}
  emb_dim: {args.emb_dim}
  depth: {args.depth}
  group_size: {args.group_size}
  num_group: {args.num_group}
  trans_dim: {args.encoder_dim}
  encoder_dims: {args.encoder_dim}
  encoder_feature: {args.encoder_feat}
  fine_dim: {args.fine_dim}
  last_dim: {args.last_dim}
  rms_norm: False
  drop_path: 0.2
  drop_out: 0.1
"""
    pod=model(name,config,args.condor)
    if(args.condor):
        print("set pod")
    if(args.eval):
        pod.loadconfig()
    config=pod.config
    cfg=re_namedtuple(yaml.safe_load(config))
    try:cand=cfg.cand
    except:pass
    batch_size=cfg.batch_size
    net=PointMODE(cfg)
    if(args.condor):
        print("set net")
    #net=PointMamba(cfg)
    pod.setnet(net.cuda())
    #summary(net,input_size=(32,2,1200),col_names=("input_size","output_size","num_params","params_percent","kernel_size","mult_adds","trainable",))
    summary(net,input_size=(batch_size,cfg.num_point,cfg.model.in_channel),depth=6,col_names=("input_size","output_size","num_params"))
    balance=args.balance
    if(not args.eval):
        print(name)
        train_dataset=h5pyData(cand,is_train=True,path=args.path,balance=balance,config=cfg)
        val_dataset=h5pyData(cand,is_val=True,path=args.path,balance=balance,config=cfg)
        train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=len(cand),shuffle=True)
        val_loader=th.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=len(cand))
        print(f"balance{train_dataset.balance} train {train_dataset.begin}-{train_dataset.end} validation {val_dataset.begin}-{val_dataset.end} ")
        print(f"train {len(train_dataset)} validation {len(val_dataset)}")
        pod.train(train_loader,val_loader,num_epoch=args.epoch)
        del train_dataset
        del val_dataset
    print(config)
    pod.loadnet()
    test_dataset=h5pyData(cand,path=args.path,balance=balance,config=cfg)
    test_loader=th.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=len(cand),shuffle=True)
    print(f"balance{test_dataset.balance} test {test_dataset.begin}-{test_dataset.end} {len(test_dataset)}")
    pod.test(test_loader)
    print("duration",datetime.datetime.now()-start_time)
