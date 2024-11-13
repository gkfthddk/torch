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
      E_dr291 = event.E_DR291
      E_dr = event.E_DR
      E_S = event.E_S
      E_C = event.E_C
      genPxyz=[event.genx,event.geny,event.genz]
      genPptp=[event.genphi,event.gentheta,event.genp]
      E_dep=event.E_dep_sim
      if(E_dep<max(max(event.fiber_e_s),max(event.fiber_e_c))):
          return None
      fiber_idx=np.array([
          list(event.fiber_idx_phi),
          list(event.fiber_idx_theta),
          list(event.fiber_e_s),
          list(event.fiber_e_c),
          list(event.fiber_idx_time),
          list(event.fiber_time),
          list(event.fiber_ix),
          list(event.fiber_iy),
          list(event.fiber_nophi),
          list(event.fiber_notheta),
          list(event.fiber_cid),
          ]).transpose()
      reco3d = np.array([
          list(event.reco3d_idx_phi),
          list(event.reco3d_idx_theta),
          list(event.reco3d_idx_time),
          list(event.reco3d_e_s),
          list(event.reco3d_e_c),
          list(event.reco3d_cid),
          ]).transpose()
      sim3d = np.array([
          list(event.sim3d_x),
          list(event.sim3d_y),
          list(event.sim3d_z),
          list(event.sim3d_en),
          list(event.sim3d_time),
          list(event.sim3d_pdg),
          list(event.sim3d_cid),
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
      
      #img_es=utils.index2img(fiber_idx[:,0],fiber_idx[:,1],fiber_idx[:,2],width=width,cen_theta=event.cen_theta)
      #img_ec=utils.index2img(fiber_idx[:,0],fiber_idx[:,1],fiber_idx[:,3],width=width,cen_theta=event.cen_theta)
      
      fiber_idx=np.array(sorted(fiber_idx, key=lambda pnt:max(pnt[2],pnt[3]),reverse=True))
      for i in range(len(fiber_idx)):
          if(fiber_idx[i,2]==0 and fiber_idx[i,3]==0):
              fiber_idx[i:]=0
              break
      reco3d=np.array(sorted(reco3d, key=lambda pnt:max(pnt[3],pnt[4]),reverse=True))
      for i in range(len(reco3d)):
          if(reco3d[i,3]==0 and reco3d[i,4]==0):
              reco3d[i:]=0
              break
      leak_gen=np.array(sorted(leak_gen, key=lambda pnt:pnt[0],reverse=True))
      if(len(leak_gen)>0):leak_tot=np.sqrt(pow(sum(leak_gen[:,1]),2)+pow(sum(leak_gen[:,2]),2)+pow(sum(leak_gen[:,3]),2))
      else:leak_tot=0.
      if(len(leak_gen)>100):leak_gen=leak_gen[:100]
      elif(len(leak_gen)==0):leak_gen=np.zeros((100-len(leak_gen),leak_gen_num_ch))
      else:leak_gen=np.concatenate([leak_gen,np.zeros((100-len(leak_gen),leak_gen_num_ch))],axis=0)
    
    #image=np.array(np.concatenate([np.expand_dims(img_es,axis=2),np.expand_dims(img_ec,axis=2)],axis=2))
    point=np.array(fiber_idx)[:3000]
    leak=np.array(leak_gen)
    reco3d=np.array(reco3d)
    
    return point,sim3d,reco3d,leak,gen_pdg,genPxyz,genPptp,E_dr291,E_dr,E_S,E_C,E_dep
    #return image,point,point_ixy,reco3d,leak,gen_pdg,genPxyz,genPptp,E_dr291,E_dr,E_S,E_C,E_dep
