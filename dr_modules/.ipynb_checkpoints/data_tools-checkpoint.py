import numpy as np
import ROOT as rt
from particle import PDGID
import dr_modules.utils as utils
from collections.abc import Sequence

def has_index(variable):
    if isinstance(variable, Sequence):
      return True
    if isinstance(variable, np.ndarray):
      return True
    return False
  
  
def h5py_form(name,data):
    dtype='float32'
    if(has_index(data)):
      shape=np.array(data).shape
    else:
      shape=tuple([])
      chunk=tuple([2048])
      return shape,dtype,chunk
    chunk=None
    if(len(shape)==3):
      chunk=(4,shape[0],shape[1],1)
    if(len(shape)==2):
      chunk=(4,3000,shape[-1])
    if(len(shape)==1):
      chunk=(2048,shape[-1])
    if(name=='ptc'):chunk=(4,64,shape[-1])
    if(name=='prt'):chunk=(256,1,shape[-1])
    if(name=='image'):chunk=(4,shape[0],shape[1],1)
    if(name=='point'):chunk=(4,shape[0],shape[1])
    if(name=='reco'):chunk=(1,shape[0],shape[1])
    if(name=='leak'):chunk=(16,shape[0],shape[1])
    
    return shape,dtype,chunk

def getk4(event, i,isjet=0,width=168):
    event.GetEntry(i)
    if(event.genp==0):
      return None
    if(len(event.fiber_idx_phi)==0):
      return None
    if(isjet==1):
      pass #FIXME
    if(isjet==0):
      gen_pdg=event.pdg
      genPxyz=[event.genx,event.genx,event.genz]
      genPptp=[event.genphi,event.gentheta,event.genp]
      fiber_idx=np.array([
          list(event.fiber_idx_phi),
          list(event.fiber_idx_theta),
          list(event.fiber_idx_e_s),
          list(event.fiber_idx_e_c),
          list(event.fiber_idx_time),
          ]).transpose()
      
      reco3d = np.array([
          list(event.reco3d_phi),
          list(event.reco3d_theta),
          list(event.reco3d_e_s),
          list(event.reco3d_e_c),
          list(event.reco3d_x),
          list(event.reco3d_y),
          list(event.reco3d_z),
          ]).transpose()
      leak_gen = np.array([
          list(event.leak_p),
          list(event.leak_px),
          list(event.leak_py),
          list(event.leak_pz),
          list(event.leak_phi),
          list(event.leak_theta),
          list(event.leak_pdg),
          list(event.leak_genstat),
          list(event.leak_simstat),
          ])
      leak_gen_num_ch=leak_gen.shape[0]
      leak_gen=leak_gen.transpose()
      
      img_es=utils.index2img(fiber_idx[:,0],fiber_idx[:,1],fiber_idx[:,2],width=width,cen_theta=event.cen_theta)
      img_ec=utils.index2img(fiber_idx[:,0],fiber_idx[:,1],fiber_idx[:,3],width=width,cen_theta=event.cen_theta)
      
      fiber_idx=np.array(sorted(fiber_idx, key=lambda pnt:max(pnt[2],pnt[3]),reverse=True))
      leak_gen=np.array(sorted(leak_gen, key=lambda pnt:pnt[0],reverse=True))
      if(len(leak_gen)>100):leak_gen=leak_gen[:100]
      elif(len(leak_gen)==0):leak_gen=np.zeros((100-len(leak_gen),leak_gen_num_ch))
      else:leak_gen=np.concatenate([leak_gen,np.zeros((100-len(leak_gen),leak_gen_num_ch))],axis=0)
    
    image=np.array(np.concatenate([np.expand_dims(img_es,axis=2),np.expand_dims(img_ec,axis=2)],axis=2))
    point=np.array(fiber_idx)
    leak=np.array(leak_gen)
    reco3d=np.array(reco3d)
    
    return image,point,reco3d,leak,gen_pdg,genPxyz,genPptp
