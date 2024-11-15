import os
import re
import numpy as np
import yaml
from collections import namedtuple

pname={'el':"$e^{-}$",'e-':"$e^{-}$",'gamma':"$\gamma$ ","gamma,e-":"$\gamma$, $e^{-}$",'pi0':"$\pi^{0}$",'pi':"$\pi^{+}$",'pi+':"$\pi^{+}$",'kaon0L':"$K^{0}_{L}$",'kaon0':"$K^{0}_{L}$",'kaon+':"$K^{+}$",'kaon':"$K^{+}$",'proton':"$p$",'neutron':"$n$",
        'uu':'u quark','dd':'d quark','ss':'s quark','cc':'c quark','gg':'Gluon','t1uu':'u quark','t1dd':'d quark','t1ss':'s quark','t1cc':'c quark','t1gg':'Gluon',
        't2uu':'u quark','t2dd':'d quark','t2ss':'s quark','t2cc':'c quark','t2gg':'Gluon','uu:dd':'Quark'}

"""
ptc=np.array([
  list(event.ptc_phi),
  list(event.ptc_theta),
  list(event.ptc_p),
  list(event.ptc_E),
  list(event.ptc_pid),
  list(event.ptc_px),
  list(event.ptc_py),
  list(event.ptc_pz),
  list(event.ptc_vx),
  list(event.ptc_vy),
  list(event.ptc_vz),
"""
def re_namedtuple(pd):
  if isinstance(pd,dict):
    for key in pd:
      if isinstance(pd[key],dict):
        pd[key] = re_namedtuple(pd[key])
    pod = namedtuple('cfg',pd.keys())
    bobo = pod(**pd)

  return bobo

def DRcorr(E_C, E_S):
    hOe_C = 0.2484
    hOe_S = 0.8342
    chi = (1.-hOe_S)/(1.-hOe_C)
    chi = 0.291
    return (E_S - chi*E_C)/(1 - chi)
def fem(E_C,E_S):
    hOe_C = 0.2484
    hOe_S = 0.8342
    return (hOe_C-hOe_S*(E_C/E_S))/((E_C/E_S)*(1-hOe_S)-(1-hOe_C))

def eta(theta):
  return -np.log((theta)/2)

def decodeID(cid):
    tobit=f'{cid:b}'
    
    nophi=int(tobit[-22:-13],2)
    
    thetabit=tobit[-13:-5]
    notheta=int(thetabit,2)
    if(pow(2,len(thetabit)-1)<notheta):
        notheta=notheta-pow(2,len(thetabit))
    
    ix=int(tobit[-39:-32],2)
    iy=int(tobit[-46:-39],2)

    is_cerenkov=bool(int(tobit[-47]))

    return nophi,notheta,ix,iy,is_cerenkov

class tower:
    def __init__(self):
      self.phi_list=[0.0,0.0222021,0.0444041,0.0666062,0.0888083,0.11101,0.133212,
        0.155414,0.177617,0.199819,0.222021,0.244223,0.266425,0.288627,0.310829,
        0.333031,0.355233,0.377435,0.399637,0.421839,0.444041,0.466243,0.488445,
        0.510648,0.53285,0.555052,0.577254,0.599456,0.621658,0.64386,0.666062,
        0.688264,0.710466,0.732668,0.75487,0.777072,0.799274,0.821477,0.843679,
        0.865881,0.888083,0.910285,0.932487,0.954689,0.976891,0.999093,1.0213,
        1.0435,1.0657,1.0879,1.1101,1.13231,1.15451,1.17671,1.19891,
        1.22111,1.24332,1.26552,1.28772,1.30992,1.33212,1.35433,1.37653,
        1.39873,1.42093,1.44313,1.46534,1.48754,1.50974,1.53194,1.55414,
        1.57635,1.59855,1.62075,1.64295,1.66516,1.68736,1.70956,1.73176,
        1.75396,1.77617,1.79837,1.82057,1.84277,1.86497,1.88718,1.90938,
        1.93158,1.95378,1.97598,1.99819,2.02039,2.04259,2.06479,2.08699,
        2.1092,2.1314,2.1536,2.1758,2.198,2.22021,2.24241,2.26461,
        2.28681,2.30902,2.33122,2.35342,2.37562,2.39782,2.42003,2.44223,
        2.46443,2.48663,2.50883,2.53104,2.55324,2.57544,2.59764,2.61984,
        2.64205,2.66425,2.68645,2.70865,2.73085,2.75306,2.77526,2.79746,
        2.81966,2.84186,2.86407,2.88627,2.90847,2.93067,2.95288,2.97508,
        2.99728,3.01948,3.04168,3.06389,3.08609,3.10829,3.13049,-3.13049,
        -3.10829,-3.08609,-3.06389,-3.04168,-3.01948,-2.99728,-2.97508,-2.95288,
        -2.93067,-2.90847,-2.88627,-2.86407,-2.84186,-2.81966,-2.79746,-2.77526,
        -2.75306,-2.73085,-2.70865,-2.68645,-2.66425,-2.64205,-2.61984,-2.59764,
        -2.57544,-2.55324,-2.53104,-2.50883,-2.48663,-2.46443,-2.44223,-2.42003,
        -2.39782,-2.37562,-2.35342,-2.33122,-2.30902,-2.28681,-2.26461,-2.24241,
        -2.22021,-2.198,-2.1758,-2.1536,-2.1314,-2.1092,-2.08699,-2.06479,
        -2.04259,-2.02039,-1.99819,-1.97598,-1.95378,-1.93158,-1.90938,-1.88718,
        -1.86497,-1.84277,-1.82057,-1.79837,-1.77617,-1.75396,-1.73176,-1.70956,
        -1.68736,-1.66516,-1.64295,-1.62075,-1.59855,-1.57635,-1.55414,-1.53194,
        -1.50974,-1.48754,-1.46534,-1.44313,-1.42093,-1.39873,-1.37653,-1.35433,
        -1.33212,-1.30992,-1.28772,-1.26552,-1.24332,-1.22111,-1.19891,-1.17671,
        -1.15451,-1.13231,-1.1101,-1.0879,-1.0657,-1.0435,-1.0213,-0.999093,
        -0.976891,-0.954689,-0.932487,-0.910285,-0.888083,-0.865881,-0.843679,-0.821477,
        -0.799274,-0.777072,-0.75487,-0.732668,-0.710466,-0.688264,-0.666062,-0.64386,
        -0.621658,-0.599456,-0.577254,-0.555052,-0.53285,-0.510648,-0.488445,-0.466243,
        -0.444041,-0.421839,-0.399637,-0.377435,-0.355233,-0.333031,-0.310829,-0.288627,
        -0.266425,-0.244223,-0.222021,-0.199819,-0.177617,-0.155414,-0.133212,-0.11101,
        -0.0888083,-0.0666062,-0.0444041,-0.0222021]
      self.theta_list=[1.55969,1.53748,1.51529,1.49314,1.47102,1.44896,1.42697,
        1.40505,1.38321,1.36147,1.33984,1.31832,1.29692,1.27566,1.25454,
        1.23357,1.21276,1.19212,1.17164,1.15135,1.13124,1.11132,1.0916,
        1.07208,1.05278,1.03369,1.01482,0.996166,0.977741,0.959546,0.941581,
        0.923851,0.906356,0.889096,0.872081,0.855311,0.838781,0.822496,0.806456,
        0.790661,0.775111,0.759806,0.744746,0.729926,0.715351,0.701021,0.686931,
        0.673081,0.659466,0.646086,0.632941,0.620026,0.607226,0.594426,0.581626,
        0.568826,0.556026,0.543226,0.530426,0.517626,0.504826,0.492026,0.479226,
        0.466426,0.453626,0.440826,0.428026,0.415226,0.402426,0.389626,0.376826,
        0.364026,0.351226,0.338426,0.325626,0.312826,0.300026,0.287226,0.274426,
        0.261626,0.248826,0.236026,0.223226,0.210426,0.197626,0.184826,0.172026,
        0.159226,0.146426,0.133626,0.120826,0.108026]
      
      self.numyx=[
        [ 0., 56., 56.], [ 1., 56., 56.], 
        [ 2., 56., 56.], [ 3., 56., 56.], [ 4., 56., 56.], [ 5., 56., 56.], [ 6., 56., 56.], 
        [ 7., 55., 56.], [ 8., 55., 56.], [ 9., 55., 55.], [10., 55., 55.], [11., 55., 55.], 
        [12., 55., 55.], [13., 55., 55.], [14., 54., 55.], [15., 54., 54.], [16., 54., 54.], 
        [17., 54., 54.], [18., 53., 54.], [19., 53., 54.], [20., 53., 53.], [21., 53., 53.], 
        [22., 52., 53.], [23., 52., 53.], [24., 52., 52.], [25., 52., 52.], [26., 51., 52.], 
        [27., 51., 52.], [28., 51., 51.], [29., 50., 51.], [30., 50., 51.], [31., 50., 50.], 
        [32., 49., 50.], [33., 49., 50.], [34., 49., 49.], [35., 48., 49.], [36., 48., 49.], 
        [37., 48., 48.], [38., 47., 48.], [39., 47., 48.], [40., 47., 47.], [41., 46., 47.], 
        [42., 46., 47.], [43., 46., 46.], [44., 45., 46.], [45., 45., 46.], [46., 45., 46.], 
        [47., 44., 45.], [48., 44., 45.], [49., 44., 45.], [50., 44., 44.], [51., 43., 44.], 
        [52., 43., 43.], [53., 43., 42.], [54., 43., 41.], [55., 43., 40.], [56., 42., 39.], 
        [57., 42., 38.], [58., 42., 37.], [59., 42., 36.], [60., 42., 35.], [61., 42., 34.], 
        [62., 41., 33.], [63., 41., 32.], [64., 41., 32.], [65., 41., 31.], [66., 41., 30.], 
        [67., 41., 29.], [68., 40., 28.], [69., 40., 27.], [70., 40., 26.], [71., 40., 25.], 
        [72., 40., 24.], [73., 40., 23.], [74., 40., 22.], [75., 40., 21.], [76., 40., 21.], 
        [77., 40., 20.], [78., 39., 19.], [79., 39., 18.], [80., 39., 17.], [81., 39., 16.], 
        [82., 39., 15.], [83., 39., 14.], [84., 39., 14.], [85., 39., 13.], [86., 39., 12.], 
        [87., 39., 11.], [88., 39., 10.], [89., 39.,  9.], [90., 39.,  8.], [91., 39.,  7.],
        ]
      self.numyx=np.array(self.numyx).astype("int")
      #notheta,min_ix,max_ix(phi),min_iy,max_iy(theta)
      self.fiphitheta=np.array([
        [0, 1, 54, 1, 54],[1, 1, 54, 1, 54],[2, 1, 54, 1, 54],[3, 1, 54, 1, 54],[4, 1, 54, 1, 54],
        [5, 1, 54, 1, 54],[6, 1, 54, 1, 54],[7, 1, 54, 0, 54],[8, 1, 54, 0, 54],[9, 0, 54, 0, 54],
        [10, 0, 54, 0, 54],[11, 0, 54, 1, 53],[12, 1, 53, 1, 53],[13, 1, 53, 1, 53],[14, 1, 53, 0, 53],
        [15, 0, 53, 0, 53],[16, 0, 53, 1, 52],[17, 0, 53, 1, 52],[18, 1, 52, 0, 52],[19, 1, 52, 0, 52],
        [20, 0, 52, 1, 51],[21, 0, 52, 1, 51],[22, 1, 51, 0, 51],[23, 1, 51, 0, 51],[24, 0, 51, 1, 50],
        [25, 0, 51, 1, 50],[26, 1, 50, 0, 50],[27, 1, 50, 1, 49],[28, 0, 50, 1, 49],[29, 1, 49, 0, 49],
        [30, 1, 49, 1, 48],[31, 0, 49, 1, 48],[32, 1, 48, 0, 48],[33, 1, 48, 1, 47],[34, 0, 48, 1, 47],
        [35, 1, 47, 0, 47],[36, 1, 47, 0, 47],[37, 0, 47, 1, 46],[38, 0, 47, 0, 46],[39, 1, 46, 0, 46],
        [40, 0, 46, 1, 45],[41, 0, 46, 0, 45],[42, 1, 45, 0, 45],[43, 0, 45, 1, 44],[44, 0, 45, 0, 44],
        [45, 1, 44, 0, 44],[46, 1, 44, 1, 43],[47, 0, 44, 0, 43],[48, 1, 43, 0, 43],[49, 1, 43, 1, 42],
        [50, 0, 43, 1, 42],[51, 1, 42, 0, 42],[52, 0, 42, 0, 42],[53, 0, 41, 0, 42],[54, 0, 40, 1, 41],
        [55, 0, 39, 1, 41],[56, 0, 38, 0, 41],[57, 0, 37, 0, 41],[58, 0, 36, 0, 41],[59, 0, 35, 1, 40],
        [60, 0, 34, 1, 40],[61, 0, 33, 1, 40],[62, 0, 32, 0, 40],[63, 0, 31, 0, 40],[64, 1, 30, 1, 39],
        [65, 1, 29, 1, 39],[66, 1, 28, 1, 39],[67, 1, 27, 1, 39],[68, 1, 26, 0, 39],[69, 1, 25, 0, 39],
        [70, 1, 24, 0, 39],[71, 0, 24, 0, 39],[72, 0, 23, 1, 38],[73, 0, 22, 1, 38],[74, 0, 21, 1, 38],
        [75, 0, 20, 1, 38],[76, 1, 19, 1, 38],[77, 1, 18, 1, 38],[78, 1, 17, 0, 38],[79, 1, 16, 0, 38],
        [80, 1, 15, 0, 38],[81, 0, 15, 0, 38],[82, 0, 14, 0, 38],[83, 0, 13, 0, 38],[84, 1, 12, 1, 37],
        [85, 1, 11, 1, 37],[86, 1, 10, 1, 37],[87, 1, 9, 1, 37],[88, 1, 8, 1, 37],[89, 0, 8, 1, 37],
        [90, 0, 7, 1, 37],[91, 0, 6, 1, 37],
      ])
      self.fiphitheta=self.fiphitheta.astype("int")
      left=0
      bottom=0
      self.theta_left=[]
      self.theta_right=[]
      self.phi_bottom=[]
      self.phi_top=[]
      for noTheta,min_ix,max_ix,min_iy,max_iy in self.fiphitheta:
          self.phi_bottom.append(bottom)
          self.theta_left.append(left)
          bottom+=int(max_ix-min_ix+1)
          left+=int(max_iy-min_iy+1)
          self.phi_top.append(bottom)
          self.theta_right.append(left)
      self.theta_left=np.array(self.theta_left)
      self.theta_right=np.array(self.theta_right)
      self.phi_bottom=np.array(self.phi_bottom)
      self.phi_top=np.array(self.phi_top)
        
    def getnumyx(self,numTheta):
      #numTheta=(1-(numTheta<0)*2)*numTheta-(numTheta<0)
      numTheta=numTheta-((numTheta<0)*2)*(numTheta+1)
      return self.numyx[:,1][numTheta],self.numyx[:,2][numTheta]
    
    def getfixy(self,numTheta):
      numTheta=numTheta-((numTheta<0)*2)*(numTheta+1)
      return self.fiphitheta[:,1][numTheta],self.fiphitheta[:,2][numTheta],self.fiphitheta[:,3][numTheta],self.fiphitheta[:,4][numTheta]
      
    def findphi(self,phi):
      diff=0
      mindiff=10
      idx=0
      if(phi>2*np.pi):
        phi=phi-2*np.pi
      
      for i in range(283):
        diff=abs(phi-self.phi_list[i])
        if(diff<mindiff):
          mindiff=diff
          idx=i
      return idx
    def findtheta(self,theta):
      diff=0
      mindiff=10
      idx=0
      for i in range(92):
        diff=abs(theta-self.theta_list[i])
        if(diff<mindiff):
          mindiff=diff
          idx=i
        #if(i!=91): FIXME what perpose?
        diff=abs(theta-(np.pi-self.theta_list[i]))
        if(diff<mindiff):
          mindiff=diff
          idx=-i-1
      return idx
    def numx_left(self,numTheta,width):
      count=0
      for i in range(1,91):
        count+=self.getnumyx(numTheta-i)[1]
        if(count>width):
          return numTheta-i
        if(numTheta-i==-92):
          return numTheta-i
    def numx_right(self,numTheta,width):
      count=0
      for i in range(1,91):
        count+=self.getnumyx(numTheta+i)[1]
        if(count>width):
          return numTheta+i
        if(numTheta+i==91):
          return numTheta+i
    def diff_theta_count(self,cenTheta,target_theta):
      cen_left=self.theta_left[cenTheta]*(cenTheta>=0)-self.theta_right[-cenTheta-1]*(cenTheta<0)
      target_left=self.theta_left[target_theta]*(target_theta>=0)-self.theta_right[-target_theta-1]*(target_theta<0) 
      return target_left-cen_left
    def diff_phi_count(self,cenPhi,target_phi):
      cen_bottom=self.phi_bottom[cenPhi]*(cenPhi>=0)-self.phi_top[-cenPhi-1]*(cenPhi<0)
      target_bottom=self.phi_bottom[target_phi]*(target_phi>=0)-self.phi_top[-target_phi-1]*(target_phi<0) 
      return target_bottom-cen_bottom
        
    def tophi(self,i):
      if(i<283):return self.phi_list[i]
      else:return self.phi_list[0]
    def totheta(self,i):
      if(i>=283):
        return self.theta_list[0]
      elif(i>=0):
        return self.theta_list[i]
      else:
        return np.pi-self.theta_list[-i]

    def get_fiberidx(self,noPhi,noTheta,ix,iy):
      if hasattr(noPhi, '__iter__'):
          if type(noPhi)==np.ndarray:
              if not "int" in noPhi.dtype.name:
                  noPhi=noPhi.astype("int")
                  noTheta=noTheta.astype("int")
                  ix=ix.astype("int")
                  iy=iy.astype("int")
          else:
              if not noPhi[0]==int and not noPhi[-1]==int:
                  noPhi=np.array([int(i) for i in noPhi])
                  noTheta=np.array([int(i) for i in noTheta])
                  ix=np.array([int(i) for i in ix])
                  iy=np.array([int(i) for i in iy])
      else:
          noPhi=int(noPhi)
          noTheta=int(noTheta)
          ix=int(ix)
          iy=int(iy)
            
      diff_phi=noPhi-0
      diff_phi=diff_phi-283*(diff_phi>=143)
      min_ix,max_ix,min_iy,max_iy=self.getfixy(noTheta)
      ix=ix-min_ix
      iy=iy-min_iy
      ix=ix+(noTheta>=0)*((max_ix-min_ix+1)-2*ix)
      iy=iy+(noTheta>=0)*((max_iy-min_iy+1)-2*iy)
    
      theta_idx=self.diff_theta_count(0,noTheta)+iy-1*(noTheta>=0)
      phi_idx=diff_phi*(max_ix-min_ix+1)+ix-1*(noTheta>=0)
    
      if hasattr(phi_idx, '__iter__'):
          if type(phi_idx)==np.ndarray:
              if not "int" in phi_idx.dtype.name:
                  phi_idx=phi_idx.astype("int")
                  theta_idx=theta_idx.astype("int")
      else:
          phi_idx=int(phi_idx)
          theta_idx=int(theta_idx)
        
      return phi_idx,theta_idx
      
def findpeak(tower_no_phi,tower_no_eta,fiber_ecor_s):
    numTower=[]
    TowerE=[]
    for i in range(len(fiber_ecor_s)):
        if(not [tower_no_phi[i],tower_no_eta[i]] in numTower):
          numTower.append([tower_no_phi[i],tower_no_eta[i]])
          TowerE.append(fiber_ecor_s[i])
        else:
          k=numTower.index([tower_no_phi[i],tower_no_eta[i]])
          TowerE[k]+=fiber_ecor_s[i]
    maxe=0
    pos=[0,0]
    for i in range(len(TowerE)):
        if(maxe<TowerE[i]):
            pos=numTower[i]
            maxe=TowerE[i]
    return pos
  
def pixelreadout(fiber_ix,fiber_iy,tower_no_phi,tower_no_eta,fiber_ecor_s,fiber_ecor_c,width=560*2,cen_phi=None,cen_theta=None):
    if(cen_phi is None and cen_theta is None):cen_phi,cen_theta=findpeak(tower_no_phi,tower_no_eta,fiber_ecor_s)
    threshold=0.0005 # 5 MeV threshold
    tower_cls=tower()
    theta_idx=[]
    phi_idx=[]
    ecor_s=[]
    ecor_c=[]
    bo=[]
    numY,numX=tower_cls.getnumyx(cen_theta)
    width_phi=int(np.ceil((width-numX)/2/numX))
    width_theta_left=tower_cls.numx_left(cen_theta,(width-numY)/2)
    width_theta_right=tower_cls.numx_right(cen_theta,(width-numY)/2)
    
    phi_edge = [cen_phi-width_phi, cen_phi+width_phi]
    left,right=False,False
    if(phi_edge[0]<0):#center over 0 left edge below 282
        phi_edge[0]=283+phi_edge[0]
        left=True
    elif(phi_edge[1]>282):#center below 282 right edge over 0
        phi_edge[1]=phi_edge[1]-283
        right=True
    for i in range(len(tower_no_eta)):
        if(fiber_ecor_s[i]<threshold and fiber_ecor_c[i]<threshold):
            continue
        noTheta=tower_no_eta[i]
        noPhi=tower_no_phi[i]
        
        diff_theta=noTheta-cen_theta #notheta increasing
        diff_phi=noPhi-cen_phi #nophi increasing
        if(noTheta<width_theta_left or noTheta>width_theta_right):continue
        if(left or right):
            if(noPhi<phi_edge[0] and noPhi>phi_edge[1]):continue
        else:
            if(abs(diff_phi)>width_phi):continue
        if(cen_phi-width_phi<0 and diff_phi>=-width_phi+283):
            diff_phi-=283
        #if(left and noPhi>=phi_edge[0]):
        #    diff_phi=noPhi-phi_edge[0]-width_phi
            """
            280,281 282 0 1 2 3
            282-0 = 282 -> -1
            281-0 = 281 -> -2
            282-1 = 281 -> -2
            281-1 = 280 -> -3
            """
        elif(cen_phi+width_phi>282 and diff_phi<=width_phi-283):
            diff_phi+=283
        #elif(right and noPhi<=phi_edge[1]):
        #    diff_phi=noPhi-phi_edge[1]+width_phi
            """
            279 280,281 282 0 1 2
            0-282 = -282 -> 1
            1-282 = -281 -> 2
            0-281 = -281 -> 2
            1-281 = -280 -> 3
            282-281= 1 1
            280-281= -1 -1
            """
        ix=fiber_ix[i]-1 #0-54 #phi
        iy=fiber_iy[i]-1 #0-54 #theta
        if(noTheta<0):
            fx=1 
            ix=ix 
            #iy=54-iy
            iy=iy # theta is opposite to eta,z,noeta

        else:
            fx=-1 
            ix=numX-ix
            #iy=iy
            iy=numY-iy # theta is opposite to eta,z,noeta
        theta_idx.append(tower_cls.diff_theta_count(cen_theta,diff_theta)+iy)
        phi_idx.append(diff_phi*numX+ix)
        ecor_s.append(fiber_ecor_s[i])
        ecor_c.append(fiber_ecor_c[i])
        bo.append([noPhi,noTheta])
    return phi_idx,theta_idx,ecor_s,ecor_c,bo
    
                   
def pixelization_tower(fiber_ix,fiber_iy,tower_no_phi,tower_no_eta,fiber_ecor_s,fiber_ecor_c,width=20,cen_phi=None,cen_theta=None):
    if(cen_phi is None and cen_theta is None):cen_phi,cen_theta=findpeak(tower_no_phi,tower_no_eta,fiber_ecor_s)
    thetabin=54
    phibin=54
    phi_edge = [cen_phi-width, cen_phi+width]
    threshold=0.005
    left,right=False,False
    if(phi_edge[0]<0):#center over 0 left edge below 282
        phi_edge[0]=283+phi_edge[0]
        left=True
    elif(phi_edge[1]>282):#center below 282 right edge over 0
        phi_edge[1]=phi_edge[1]-283
        right=True
    theta_idx=[]
    phi_idx=[]
    ecor_s=[]
    ecor_c=[]
    bo=[]
    for i in range(len(tower_no_eta)):
        if(fiber_ecor_s[i]<threshold and fiber_ecor_c[i]<threshold):
            continue
        noTheta=tower_no_eta[i]
        noPhi=tower_no_phi[i]
        ix=fiber_ix[i]-1 #0-54 #phi
        iy=fiber_iy[i]-1 #0-54 #theta
        if(noTheta<0):
            fx=1 
            ix=ix 
            #iy=54-iy
            iy=iy # theta is opposite to eta,z,noeta

        else:
            fx=-1 
            ix=54-ix
            #iy=iy
            iy=54-iy # theta is opposite to eta,z,noeta

        diff_theta=noTheta-cen_theta #notheta increasing
        diff_phi=noPhi-cen_phi #nophi increasing
        if(abs(diff_theta)>width):continue
        if(left or right):
            if(noPhi<phi_edge[0] and noPhi>phi_edge[1]):continue
        else:
            if(abs(diff_phi)>width):continue
        if(left and noPhi>=phi_edge[0]):
            diff_phi=noPhi-phi_edge[0]-width
            """
            280,281 282 0 1 2 3
            282-0 = 282 -> -1
            281-0 = 281 -> -2
            282-1 = 281 -> -2
            281-1 = 280 -> -3
            """
        elif(right and noPhi<=phi_edge[1]):
            diff_phi=noPhi-phi_edge[1]+width
            """
            279 280,281 282 0 1 2
            0-282 = -282 -> 1
            1-282 = -281 -> 2
            0-281 = -281 -> 2
            1-281 = -280 -> 3
            282-281= 1 1
            280-281= -1 -1
            """
        theta_idx.append(diff_theta*thetabin+iy)
        phi_idx.append(diff_phi*phibin+ix)
        ecor_s.append(fiber_ecor_s[i])
        ecor_c.append(fiber_ecor_c[i])
        bo.append([noPhi,noTheta])
    return phi_idx,theta_idx,ecor_s,ecor_c,bo
  
  
def getfiles(key,begin=0,end=100,path="npzs"):#find npz file with key_<begin> ~ key_<end>
    files_with_key=[]
    for f in os.listdir(path):
        if(key in f):
            files_with_key.append(f)
    files=[]
    for i in range(begin,end):
        if(os.path.isfile(path+"/"+f'{key}_{i}.npz')):
          files.append(path+"/"+f'{key}_{i}.npz')
    return files

def index2img(phiindex,thetaindex,energy,width=168,from_width=560,cen_theta=0,**kwargs):
    tower_cls=tower()
    #edge=int(from_width*2.1/2-width/2)#FIXME
    edge=int(width/2)-int(tower_cls.getnumyx(cen_theta)[1]/2)
    phiindex=np.array(phiindex)+edge
    thetaindex=np.array(thetaindex)+edge
    phiindex=np.around(phiindex)
    thetaindex=np.around(thetaindex)
    phiindex=phiindex.astype(int)
    thetaindex=thetaindex.astype(int)
    _img=np.zeros((width,width))
    img_shape=_img.shape
    for i in range(len(phiindex)):
        if(phiindex[i]<img_shape[0] and phiindex[i]>=0):
          if(thetaindex[i]<img_shape[0] and thetaindex[i]>=0):
            try:_img[thetaindex[i],phiindex[i]]+=energy[i]
            except:print(thetaindex[i],phiindex[i],energy[i])
            _img[thetaindex[i],phiindex[i]]+=energy[i]
    return _img

def pooltable(pattern=r'drt4_([A-Za-z-+0]+)_([A-Za-z-+0]+)_img_h5_p([0-9]+)_nout.npz',pids=['e-','gamma','pi0','pi+','kaon0','kaon+','neutron','proton']):
    table={}
    for i in range(len(pids)):
        for j in range(i+1,len(pids)):
            table['{}_{}'.format(pids[i],pids[j])]=[]

    #drt4_neutron_proton_img_h5_p1_nout.npz

    for f in os.listdir("drbox/"):
        span=re.search(pattern,f)
        if(span):
            p1=span.group(1)
            p2=span.group(2)
            pool=int(span.group(3))
            key='{}_{}'.format(p1,p2)
            key_r='{}_{}'.format(p2,p1)
            if(key in table):
                table[key].append([pool,key])
            if(key_r in table):
                table[key_r].append([pool,key])
    for key in table:
        table[key]=sorted(table[key])

    pad="".join(["{0: <19}".format(k) for k in ["vs"]+pids])
    for i in range(len(pids)):
        pad+="\n{0: <19}".format(pids[i])
        for j in range(len(pids)):
            key='{}_{}'.format(pids[i],pids[j])
            key_r='{}_{}'.format(pids[j],pids[i])
            if(key_r) in table:
                key=key_r
            cell="-"
            if(key in table):
                cell=" ".join([str(k[0]) for k in table[key]])

            pad+="{0: <19}".format(cell)
    print(pad)
    return table

