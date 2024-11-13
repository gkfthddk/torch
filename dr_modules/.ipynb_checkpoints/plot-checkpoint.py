import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import tqdm
import re
import h5py
from matplotlib.ticker import FormatStrFormatter
import dr_modules.utils as utils
pname={'el':"$e^{-}$",'e-':"$e^{-}$",'gamma':"$\gamma$ ",'pi0':"$\pi^{0}$",'pi':"$\pi^{+}$",'pi+':"$\pi^{+}$",'kaon0L':"$K^{0}_{L}$",'kaon0':"$K^{0}_{L}$",'kaon+':"$K^{+}$",'kaon':"$K^{+}$",'proton':"$p$",'neutron':"$n$",
        'uu':'u quark','dd':'d quark','ss':'s quark','cc':'c quark','gg':'Gluon','t1uu':'u quark','t1dd':'d quark','t1ss':'s quark','t1cc':'c quark','t1gg':'Gluon',
       't2uu':'u quark','t2dd':'d quark','t2ss':'s quark','t2cc':'c quark','t2gg':'Gluon','uu:dd':'Quark'}
def loadauc(path):
    load=np.load(path,allow_pickle=True)
    y=load['y']
    if(len(y.shape)==2):
        y=y[:,0]
    target= y
    pred=load['p'][:,0]
    fpr, tpr, _ = roc_curve(target,pred)
    return auc(fpr, tpr)
"""list(event.prt_phi),
          list(event.prt_theta),
          list(event.prt_E),
          list(event.prt_pid),
          list(event.prt_px),
          list(event.prt_py),
          list(event.prt_pz),
          ]).transpose()"""


class boxloader:
    def __init__(self):
        self.load=[]
        self.label=[]
        self.target=[]
        self.pred=[]
        self.fpr=[]
        self.tpr=[]
        self.auc=[]
    def add(self,path,label=None):
        self.load.append(np.load(path,allow_pickle=True))        
        if(label is None):
            label=path.split("/")[1].split("out.")[0]
        cand=self.load[len(self.label)]['cand']
        span=re.search(r'_p([0-9]+)_',path)
        pool=None
        if(span):
          pool=int(span.group(1))
          pool=f"{int(168/pool)}x{int(168/pool)}"
        label_buf=[pname[cand[0].split("_")[0]],pname[cand[1].split("_")[0]],label,pool]
        self.label.append(label_buf)
        y=self.load[-1]['y']
        if(len(y.shape)==2):
            y=y[:,0]
        self.target.append(y)
        self.pred.append(self.load[-1]['p'][:,0])
        fpr, tpr, _ = roc_curve(self.target[-1], self.pred[-1])
        self.fpr.append(fpr)
        self.tpr.append(tpr)
        self.auc.append(auc(fpr, tpr))
    def response(self,title,num=-1,fs=5,density=True,dpi=100,logscale=False,figsize=1.6):
        
        fig,B=plt.subplots(ncols=1,nrows=1,dpi=dpi,figsize=(fs*figsize,fs))
        count=0
        for i in range(len(self.load)):
            if(num!=-1 and num!=i):continue
            y=self.load[i]['y']
            if(len(y.shape)==2):
                y=y[:,0]
            p=self.load[i]['p'][:,0]
            B.hist(p[np.where(y==1)[0]],bins=30,alpha=0.5,label=self.label[i][0],density=density,color='C{}'.format(count*2))
            B.hist(p[np.where(y==0)[0]],bins=30,alpha=0.5,label=self.label[i][1],density=density,color='C{}'.format(count*2+1))
            count+=1
                
        B.set_xlabel("Deep learning model response",fontsize=fs*3.5,labelpad=fs*2)
        B.set_ylabel("dN/dx",fontsize=fs*4)
        if(logscale):B.set_yscale('log')
        a1,a2,b1,b2=B.axis()
        B.axis((0,1,b1,b2))
        B.tick_params(axis = 'x', labelsize =fs*3.5)
        B.tick_params(axis = 'y', labelsize =fs*3.5)
        B.legend(loc=9,fontsize=fs*4,shadow=False,framealpha=0.8,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        B.set_title(title,fontsize=fs*3.8,y=1.03)
    def roc(self,title,legend='vs',mineff=0,num=-1,fs=5,density=True,dpi=100,figsize=1.6):
        fig,B=plt.subplots(ncols=1,nrows=1,dpi=dpi,figsize=(fs*figsize,fs))
        count=0
        for i in range(len(self.load)):
            if(num!=-1 and num!=i):continue
            #y=self.load[i]['y']
            #target=self.target[i]
            #pred=self.pred[i]
            if(legend=='vs'):
                label="{:<5} vs {:<5}".format(self.label[i][0],self.label[i][1])
            elif(legend=='pool'):
                label=self.label[i][3]
            else:
                label=self.label[i][2]
            auc_label="{} : {:0.3f}".format(label,int(self.auc[i]*1000)/1000.)
            #auc_label="{:<5} vs {:<5} : {:0.3f}".format(siglabel,baklabel,int(buf_auc*1000)/1000.)
            B.plot(self.tpr[i],1-self.fpr[i],label=auc_label,lw=fs*0.5,alpha=0.65,color='C{}'.format(count*1))
            count+=1
        if(legend=='vs'):
            ax_legend=B.legend(loc=3,fontsize=fs*3.5,title="      $\it{sig}$ vs $\it{bkg}$ :  AUC",title_fontsize=fs*3.,shadow=True,framealpha=1.,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        elif(legend=='pool'):
            ax_legend=B.legend(loc=3,fontsize=fs*3.,title="  Resolution :  AUC",title_fontsize=fs*3.,shadow=True,framealpha=1.,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        else:
            ax_legend=B.legend(loc=3,fontsize=fs*3.5,title="  Model type :  AUC",title_fontsize=fs*3.,shadow=True,framealpha=1.,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        #legend._legend_box.align = "right"
        #shift = max([text.get_window_extent().width for text in legend.get_texts()])
        #for text in legend.get_texts():
        #    text.set_ha('right')
            #print(text.get_window_extent().width)
        #    text.set_position((fs*38,0))
        #   text.set_color("red")
        B.axis((mineff,1,mineff,1))
        B.set_xlabel("Signal efficiency",fontsize=fs*3.5,labelpad=fs*2)
        B.set_ylabel("Background rejection",fontsize=fs*3.5)
        #B.set_xticks([0.8,0.85,0.9,0.95,1.0])
        #B.set_yticks([0.8,0.85,0.9,0.95,1.0])
        B.tick_params(axis = 'x', labelsize =fs*3.5)
        B.tick_params(axis = 'y', labelsize =fs*3.5)
        B.grid(lw=fs*0.25,alpha=0.6)
        B.set_title(title,fontsize=fs*3.8,y=1.03)
        B.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    def getcut(self,title,legend='vs',num=-1,fs=5,density=True,dpi=100):
        fig,B=plt.subplots(ncols=1,nrows=1,dpi=dpi,figsize=(fs*1.6,fs))
        count=0
        for i in range(len(self.load)):
            if(num!=-1 and num!=i):continue
            fpr, tpr= self.fpr[i], self.tpr[i]
            tnr=1-fpr
            delta,dx=[],[]
            mm,mran=[],[]
            res=100
            for l in range(res,len(tpr),res):
                dd=(tnr[l]-tnr[l-res])/(tpr[l]-tpr[l-res])
                dx.append(tpr[l])
                #dx.append(l/len(tpr))
                #delta.append((tnr[l]+tpr[l]))
                delta.append(abs(((tnr[l]+tpr[l])-(tnr[l-res]+tpr[l-res]))/(tpr[l]-tpr[l-res])))
                mm.append(np.sqrt(np.mean([pow(tnr[l],2),pow(tpr[l],2)])))
                mran.append(l)
            nidx=mran[np.where(delta==min(delta))[0][0]]
            midx=mran[np.where(mm==max(mm))[0][0]]
            sss=(1-tpr)*(1-tpr)+(1-tnr)*(1-tnr)
            nss=np.where(sss==min(sss))
            #if(verbose):print("grad",tpr[nidx],tnr[nidx],"rms",tpr[midx],tnr[midx])
            #if(verbose):print(tpr[nss],tnr[nss])
            if(legend=='vs'):
                label="{:<5} vs {:<5}".format(self.label[i][0],self.label[i][1])
            else:
                label=self.label[i][2]
            auc_label="{} : {:0.3f}".format(label,int(self.auc[i]*1000)/1000.)
            B.plot(dx,delta,label=auc_label,alpha=0.8)
        B.set_xlabel("Signal efficiency",fontsize=fs*3.5,labelpad=fs*2)
        B.set_ylabel("$\\Delta$rejection",fontsize=fs*3.5)
        B.tick_params(axis = 'x', labelsize =fs*3.5)
        B.tick_params(axis = 'y', labelsize =fs*3.5)
        B.grid(lw=fs*0.25,alpha=0.6)
        B.axis((0.8,1,0,10))
        B.set_title("Best cut",fontsize=fs*3.8,y=1.03)
    def blockauc(self,title,legend='vs',mineff=0,num=-1,fs=5,density=True,dpi=100,coeff=[2,.2,-1.05],threshold=0.7,verbose=False):
        cmap1 = matplotlib.cm.get_cmap("Oranges")
        cmap2 = matplotlib.cm.get_cmap("Blues")
        pids=['e-','gamma','pi0','pi+','kaon0','kaon+','neutron','proton']
        table={}
        pattern=r'drt4_([A-Za-z-+0]+)_([A-Za-z-+0]+)_img_h5_p([0-9]+)_nout.npz'
        for i in range(len(pids)):#1
            for j in range(i+1,len(pids)):
                table['{}_{}'.format(pids[i],pids[j])]=[]
        for i in range(len(self.load)):
            filename=self.load[i].zip.filename
            span=re.search(pattern,filename)
            p1=span.group(1)
            p2=span.group(2)
            key='{}_{}'.format(p1,p2)
            key_r='{}_{}'.format(p2,p1)
            if(key in table):
                table[key].append(self.auc[i])
            if(key_r in table):
                table[key_r].append(self.auc[i])
        aucs=block=np.zeros((len(pids),len(pids)))
        block=np.zeros((len(pids),len(pids),4))
        pad="".join(["{0: <19}".format(k) for k in ["vs"]+pids])
        for i in range(len(pids)):#2
            pad+="\n{0: <19}".format(pids[i])
            for j in range(len(pids)):
                key='{}_{}'.format(pids[i],pids[j])
                key_r='{}_{}'.format(pids[j],pids[i])
                if(key_r) in table:
                    key=key_r
                cell="-"
                if(key in table):
                    cell=" ".join(["{:0.3f}".format(k) for k in table[key]])
                    for k in table[key]:
                        fill_color=cmap1(k*coeff[0]+coeff[1]*np.sin((k-0.5)*np.pi*2)+coeff[2])
                        block[i,j] = fill_color
                        aucs[i,j] = k
                        block[j,i] = fill_color
                        aucs[j,i] = k
                pad+="{0: <19}".format(cell)
        if(verbose):print(pad)
        fig,B=plt.subplots(ncols=1,nrows=1,dpi=dpi,figsize=(fs*len(pids)/4,fs*len(pids)/4))
        B.set_xticks(range(len(pids)))
        B.set_xticklabels([pname[pid] for pid in pids],fontsize=fs*5)
        B.set_yticks(range(len(pids)))
        B.set_yticklabels([pname[pid] for pid in pids],fontsize=fs*5)
        B.set_xticks(np.arange(-0.5,len(pids),1),minor=True)
        B.set_yticks(np.arange(-0.5,len(pids),1),minor=True)
        B.grid(which='minor', color='w', linestyle='-', linewidth=1)
        B.tick_params(which='major',top=True,right=True,labeltop=True,labelright=True)
        B.tick_params(which='minor', bottom=False, left=False)
        
        B.imshow(block,origin='lower')
        for i in range(len(pids)):#3
            for j in range(len(pids)):
                if(aucs[j,i]==0):
                  pass
                elif(aucs[j,i]<threshold):
                  B.text(i-0.37,j-0.05,"{:0.3f}".format(int(aucs[j,i]*1000)/1000.),color="black",fontsize=fs*3.3)
                else:
                  B.text(i-0.37,j-0.05,"{:0.3f}".format(int(aucs[j,i]*1000)/1000.),color="white",fontsize=fs*3.3)
        B.set_title(title,fontsize=fs*3.8)
        
    def perform(self,title=None,res=10,key='genphi',num=-1,tag=-1,legend='pool',fs=5,density=True,dpi=100,xrange=None,xlabel=None,ylabel=None):
        if(title is None):
            title=key
        fig,B=plt.subplots(ncols=1,nrows=1,dpi=dpi,figsize=(fs*1.7,fs))
        for i in range(len(self.load)):
            hf=[]
            testidx=self.load[i]['testidx']
            #-- delete this section when testidx is fixed 
            if(testidx[0][0]==1,testidx[2][0]==1,testidx[2][1]==0):
              idx_buf=[]
              count=0
              check=False
              for j in range(len(testidx)):
                if(testidx[j][0]==0 and check==False):
                  check=True
                  count=0
                idx_buf.append([1-testidx[j][0],count])
                count+=1
              testidx=np.array(idx_buf).astype(int)
            #--
            for h5path in self.load[i]['h5path']:
                hf.append(h5py.File(h5path.replace("_test",""),'r'))
            v=[]
            target=self.target[i]
            pred=self.pred[i]
            entries=len(testidx)
            if(num!=-1):
                testidx,target,pred=shuffle(testidx,target,pred)
                entries=min(num,entries)
            for j in tqdm.trange(entries):
                idx=testidx[j]
                if(len(hf[idx[0]][key].shape)==2):
                    if(tag==456):
                        v.append(np.sqrt(np.sum([hf[idx[0]][key][idx[1]][0][k]**2 for k in [4,5,6]])))
                    else:
                        v.append(hf[idx[0]][key][idx[1]][tag])
                else:
                    v.append(hf[idx[0]][key][idx[1]])
            
            minv=min(v)
            binv=(max(v)-min(v))/res
            v=np.array(v)
            x,y=[],[]
            for j in range(res):
                app=np.where((v>minv+binv*j) & (v<minv+binv*(j+1)))[0]
                fpr, tpr, _ = roc_curve(target[app], pred[app])
                x.append(minv+binv*(j+0.5))
                y.append(auc(fpr, tpr))
            if(legend=='pool'):
              label=self.label[i][3]
            else:
              label=self.label[i][2]
            B.plot(x,y,label=label,alpha=0.8)
            
        #if(not xrange is None):
        #    a1,a2,b1,b2=B.axis()
        #    B.axis((xrange[0],xrange[1],b1,b2))
        
        B.tick_params(axis = 'x', labelsize =fs*3.5)
        B.tick_params(axis = 'y', labelsize =fs*3.5)
        B.grid(lw=fs*0.25,alpha=0.6)
        #if(not xrange is None):
        #    a1,a2,b1,b2=B.axis()
        #    B.axis((xrange[0],xrange[1],b1,b2))
        if(legend=='pool'):
          ax_legend=B.legend(loc=8,fontsize=fs*3.,title=" Resolution ",title_fontsize=fs*3.,shadow=True,framealpha=1.,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        else:
          ax_legend=B.legend(loc=8,fontsize=fs*3.,title=" label ",title_fontsize=fs*3.,shadow=True,framealpha=1.,edgecolor="grey",facecolor=(0.999, 0.998, 0.99))
        B.set_title(title,fontsize=fs*3.8,y=1.03)
        if(not xlabel is None):
          B.set_xlabel(xlabel,fontsize=fs*3.5,labelpad=fs*2)
        else:
          if key=="prt":
              if(tag==2):
                  xlabel="Particle Energy (GeV)"
              if(tag==456):
                  xlabel="Particle Momentum (GeV)"
          elif key=="genphi":
              xlabel="Particle direction $\phi$"
          elif key=="gentheta":
              xlabel="Particle direction $\\theta$"
          else:
              xlabel="None"
          B.set_xlabel(xlabel,fontsize=fs*3.5,labelpad=fs*2)
        if(not ylabel is None):B.set_ylabel(ylabel,fontsize=fs*3.5)
        else:B.set_ylabel("AUC",fontsize=fs*3.5)