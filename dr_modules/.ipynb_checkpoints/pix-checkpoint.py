import os
import numpy as np
from numpy.ma import masked_array
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc
import h5py

pname={'el':"$e^{-}$",'e-':"$e^{-}$",'gamma':"$\gamma$ ",'pi0':"$\pi^{0}$",'pi':"$\pi^{+}$",'pi+':"$\pi^{+}$",'kaon0L':"$K^{0}_{L}$",'kaon0':"$K^{0}_{L}$",'kaon+':"$K^{+}$",'kaon':"$K^{+}$",'proton':"$p$",'neutron':"$n$",
        'uu':'u quark','dd':'d quark','ss':'s quark','cc':'c quark','gg':'Gluon','t1uu':'u quark','t1dd':'d quark','t1ss':'s quark','t1cc':'c quark','t1gg':'Gluon',
       't2uu':'u quark','t2dd':'d quark','t2ss':'s quark','t2cc':'c quark','t2gg':'Gluon','uu:dd':'Quark'}

def getimage(path,num=0,en=[],getp=False,verbose=False):
    hf=h5py.File(path,'r')
    p=None
    if("genPptp" in  hf.keys()):
      genp=hf['genPptp'][:,2]
    elif("prt" in  hf.keys()):
      genp=np.sqrt(np.sum([hf['prt'][:,0,k]**2 for k in [4,5,6]],axis=0))
    if(en!=[] and len(en)==2):
        idx=np.where((genp>=en[0]) & (genp<en[1]))[0]
    else:
        idx=np.arange(len(genp))
    if(num==1):
        num=np.random.randint(0,len(idx))
        if(getp):
            p=genp[idx[num]]
        if(verbose):print(path,idx[num])
        scint=hf['image'][idx[num],:,:,0]#Channel last
        ceren=hf['image'][idx[num],:,:,1]
    elif(num==0):
        scint=np.mean(hf['image'][idx[:],:,:,0],axis=0)
        ceren=np.mean(hf['image'][idx[:],:,:,1],axis=0)
    elif(num>0):
        scint=np.mean(hf['image'][idx[:num],:,:,0],axis=0)
        ceren=np.mean(hf['image'][idx[:num],:,:,1],axis=0)
    else:
        num=-num
        if(num>=len(hf['image'])):
          raise "num should be smaller than {}".format(len(hf['image']))
        if(getp):
            p=genp[num]
        scint=hf['image'][num,:,:,0]
        ceren=hf['image'][num,:,:,1]
    return scint,ceren,p
def applypool(img,stride=2):
    padding=int(np.ceil((stride-img.shape[0]%stride)/2))
    if(img.shape[0]%stride!=0):img=np.pad(img, ((padding,padding),(padding,padding)), 'constant', constant_values=0)
    applypoolbin=int(img.shape[0]/stride)
    img_p=np.zeros((applypoolbin,applypoolbin))
    for i in range(applypoolbin):
        for j in range(applypoolbin):
            img_p[i,j]=img[i*stride:(i+1)*stride,j*stride:(j+1)*stride].sum()
    return img_p
def zoom(pad,cut=0,percent=1,position=[0.5,0.5]):
    pad=pad[cut:pad.shape[0]-cut,cut:pad.shape[1]-cut]
    xsize=pad.shape[1]
    ysize=pad.shape[0]
    position=[int(position[0]*xsize),int(position[1]*ysize)]
    xsize=int(xsize*percent)
    ysize=int(ysize*percent)
    position=[int(position[0]-xsize/2),int(position[1]-ysize/2)]
    return(pad[position[1]:position[1]+ysize,position[0]:position[0]+xsize])

def getcmap(colormap="gist_heat_r"):
    #colormap="viridis"
    cmap = matplotlib.cm.get_cmap(colormap)
    rgba = cmap(1.0)
    rgba = [round(np.mean(rgba[:3]))]*3+[1]
    cmap = matplotlib.cm.get_cmap('gist_heat_r', int(256/0.4))
    newcolors = cmap(np.linspace(0, 0.4, 256))
    scintcmap = matplotlib.colors.ListedColormap(newcolors)
    cmap = matplotlib.cm.get_cmap('gist_heat_r', int(256/0.4))
    newcolors = cmap(np.linspace(0, 0.4, 256))
    for i in range(len(newcolors)):
            r,g,b,t=newcolors[i]
            r,g,b=[1-0.8*(1-r),1-0.8*(1-g),1-0.8*(1-b)]
            newcolors[i]=[b,g,r,t]
    cerencmap = matplotlib.colors.ListedColormap(newcolors)
    return scintcmap,cerencmap

class canvas():
    def __init__(self, ncols, nrows,figsize=None,colormap="gist_heat_r",dpi=100,fs=10):
        if(figsize is None):figsize=(ncols*fs,nrows*5.5*fs/6)
        plt.subplots_adjust(wspace=0.001*fs)
        self.fig,B=plt.subplots(ncols=ncols,nrows=nrows,figsize=figsize,dpi=dpi)
        if(ncols==1 and nrows==1):
            self.B=np.array([[B]])
        elif(ncols==1 or nrows==1):
            self.B=np.array([B])
        else:
            self.B=B
        plt.close()
        self.ncols,self.nrows=ncols,nrows
        self.fs=fs
        self.scint_cmap,self.ceren_cmap=getcmap(colormap)
        self.colormap=colormap
        cmap = matplotlib.cm.get_cmap(colormap)
        rgba = cmap(1.0)
        self.rgba = [round(np.mean(rgba[:3]))]*3+[1]
        self.scint_image=[]
        self.ceren_image=[]
        self.p=[]
        self.title=[]
        self.key=[]
    def load(self,path,title,**kwargs):
        scint,ceren,p=getimage(path,**kwargs)
        key=path.split("/")[-1]
        pid=key.split("_")[0]
        title=title.replace("{pname}",pname[pid])
        self.title.append(title)
        self.key.append(key)
        self.scint_image.append(scint)
        self.ceren_image.append(ceren)
        self.p.append(p)
        return len(self.title)-1
    def setticks(self,ax,cut,pool):
        if(cut==56):
            ax.set_xticks([1,28,55])
            ax.set_xticklabels([int(1/pool),int(28/pool),int(55/pool)])
            #ax.set_xticklabels([-0.01,0,0.01])
            ax.set_yticks([int(1/pool),int(28/pool),int(55/pool)])
            ax.set_yticklabels([1,28,55])
            #ax.set_yticklabels([1.57,1.56,1.55])
        if(cut==0):
            ax.set_xticks([0,int(56/pool),int(112/pool),int(168/pool)])
            ax.set_xticklabels([0,int(56),int(112),int(168)])
            #ax.set_xticklabels([-0.03,-0.01,0.01,0.03])
            ax.set_yticks([0,int(56/pool),int(112/pool),int(168/pool)])
            ax.set_yticklabels([0,int(56),int(112),int(168)])
            #ax.set_yticklabels([1.59,1.57,1.55,1.53])
        ax.tick_params(labelsize=1.5*self.fs)
        ax.set_xlabel('$\phi$',fontsize=2.5*self.fs)
        ax.set_ylabel('$\\theta$',fontsize=2.5*self.fs)
    def setcolorbar(self,im,ax,title):
        cbb=self.fig.colorbar(im, ax=ax)
        cbb.set_label('Energy(GeV)',fontsize=self.fs)
        cbb.ax.set_title(title,fontsize=self.fs)
    def overlapat(self,k,row=0,col=0,cut=0,getp=False,text=None,colorbar=False):
        scint_image=zoom(self.scint_image[k],cut=cut)
        ceren_image=zoom(self.ceren_image[k],cut=cut)
        v1a = masked_array(scint_image,scint_image==0)
        v1b = masked_array(ceren_image,ceren_image==0)
        pid=self.key[k].split("_")[0]
        text=text.replace("{pname}",pname[pid])
        #pa = self.B[row,col].imshow(scint_norm,interpolation='None',cmap=scintcmap,origin="lower")
        #pb = self.B[row,col].imshow(v1b,interpolation='None',cmap=cerencmap,origin="lower")
        pa=self.B[row,col].pcolor(v1a, edgecolors='w', linewidth=0.2,cmap=self.scint_cmap)
        pb=self.B[row,col].pcolor(v1b, edgecolors='w', linewidth=0.2,cmap=self.ceren_cmap)
        self.setticks(self.B[row,col],cut)
        if(colorbar):
          cbb = plt.colorbar(pb,shrink=0.7,pad=0.06,ax=self.B[row,col])
          cba = plt.colorbar(pa,shrink=0.7,pad=0.05,ax=self.B[row,col])
          cbb.set_label('Energy(GeV)',fontsize=1.25*self.fs)
          cba.ax.tick_params(labelsize=1.3*self.fs)
          cbb.ax.tick_params(labelsize=1.3*self.fs)
          cbb.ax.set_anchor((-0.4,0.5))
          cba.ax.set_title("Scintilation",fontsize=1.2*self.fs)
          cbb.ax.set_title("Cerenkov",fontsize=1.2*self.fs)
        self.B[row,col].set_title(self.title[k],fontsize=2.5*self.fs)
        self.B[row,col].grid(which='minor', color='b', linestyle='-', linewidth=2)
        self.B[row,col].set_aspect('equal')
        if(not text is None):self.B[row,col].text(scint_image.shape[0]*0.04,scint_image.shape[1]*0.9,text,fontsize=self.fs*2.5,color=self.rgba)
        if(getp):
          if self.p[k] is None:
            ptext="None"
          else:
            ptext="{:0.1f} GeV".format(self.p[k])
          self.B[row,col].text(scint_image.shape[0]*0.68,scint_image.shape[1]*0.93,ptext,fontsize=self.fs*2.,color=self.rgba)
        plt.close()
    def imshowat(self,k,row=-1,col=0,text=None,colorbar=False,isceren=False,cut=0,getp=False,pool=1):
        fs_buf=self.fs
        self.fs=fs_buf*1.4
        pid=self.key[k].split("_")[0]
        text=text.replace("{pname}",pname[pid])
        if(row==-1):
            rows=[0,1]
            iscerens=[False,True]
        else:
            rows=[row]
            iscerens=[isceren]
        for row,isceren in zip(rows,iscerens):
            if(isceren):
                image=zoom(self.ceren_image[k],cut=cut)
                bartitle="Cerenkov"
                cmap=self.ceren_cmap
            else:
                image=zoom(self.scint_image[k],cut=cut)
                bartitle="Scintilation"
                cmap=self.scint_cmap
            self.B[row,col].clear()
            image=applypool(image,pool)
            im=self.B[row,col].imshow(image,origin="lower",cmap=cmap)
            self.setticks(self.B[row,col],cut,pool)
            if(colorbar):self.setcolorbar(im,self.B[row,col],bartitle)
            if(not text is None):self.B[row,col].text(image.shape[0]*0.04,image.shape[1]*0.9,text,fontsize=self.fs*2.5,color=self.rgba)
            if(getp):self.B[row,col].text(image.shape[0]*0.68,image.shape[1]*0.93,"{:0.1f} GeV".format(self.p[k]),fontsize=self.fs*2.,color=self.rgba)
            self.B[row,col].set_title(self.title[k],fontsize=2.5*self.fs)
        self.fs=fs_buf
        

    