import numpy as np
import random
import h5py
import os
import tqdm
import multiprocessing
"""
point 
  phi theta e_s e_c time
  self.xmax = [1100, 3800, 15, 25, 300] 
reco3d 
  phi theta time e_s e_c cid
  self.xmax = [1100, 3800, 15, 25, 300] 
sim3d 
  x y z en time pdg cid
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


def pool_time(X,pool=1,num_point=None):
    if(num_point is None):
        num_point=len(X)
    X_buf=np.zeros((num_point,5))
    count=0
    for i in range(len(X)):
        check=False
        if(X[i][2]==0 and X[i][3]==0):continue
        time_bin=int(X[i][4]/10)
        phi_bin=int(X[i][0]/pool)
        theta_bin=int(X[i][1]/pool)
        for j in range(len(X_buf)):
            if(X_buf[j][2]==0 and X_buf[j][3]==0):
                break
            if(X_buf[j][0]==phi_bin and X_buf[j][1]==theta_bin and X_buf[j][4]==time_bin):
                X_buf[j][2]+=X[i][2]
                X_buf[j][3]+=X[i][3]
                check=True
                break
        if(not check and count<num_point):
            X_buf[count][0]=phi_bin
            X_buf[count][1]=theta_bin
            X_buf[count][4]=time_bin
            X_buf[count][2]+=X[i][2]
            X_buf[count][3]+=X[i][3]
            count+=1
    return X_buf
writekey=[
    "E_dep",
    "Leak_sum",
    "GenParticles.PDG",
    "GenParticles.momentum.p","GenParticles.momentum.theta","GenParticles.momentum.phi",
    "Leakages.PDG",
    "Leakages.momentum.p","Leakages.momentum.theta","Leakages.momentum.phi",
    'SimCalorimeterHits.cellID','SimCalorimeterHits.energy',
    'SimCalorimeterHits.position.x','SimCalorimeterHits.position.y','SimCalorimeterHits.position.z',
    'DRcalo2dHits.cellID','DRcalo2dHits.energy',
    'DRcalo2dHits.time',
    'DRcalo2dHits.type',
    "DRcalo2dHits.noPhi","DRcalo2dHits.noTheta",
    "DRcalo2dHits.ix","DRcalo2dHits.iy",
    'DRcalo2dHits.phi_idx','DRcalo2dHits.theta_idx',
    'DRcalo3dHits.cellID','DRcalo3dHits.energy',
    'DRcalo3dHits.time',
    'DRcalo3dHits.type',
    "DRcalo3dHits.noPhi","DRcalo3dHits.noTheta",
    "DRcalo3dHits.ix","DRcalo3dHits.iy",
    'DRcalo3dHits.phi_idx','DRcalo3dHits.theta_idx',
    ]

def read_h5(read_path,output,proc):
    count=0
    if(not os.path.isfile(read_path)):
        output.put((-1,{},proc))
        return 0
    read=h5py.File(read_path,'r')
    k=0
    for i in range(len(read['GenParticles.PDG'])):
      gen_pdg=read['GenParticles.PDG'][i][0]
      if(gen_pdg==0):
          break
      k+=1
    datas={}
    for key in writekey:
      print(key)
      datas[key]=read[key][:k]
      #gen_p=read['GenParticles.momentum.p'][i][0]
      #gen_phi=read['GenParticles.momentum.phi'][i][0]
      #gen_theta=read['GenParticles.momentum.theta'][i][0]
    E_dep=np.sum(read['SimCalorimeterHits.energy'][:k],axis=1)
    Leak_sum=np.sum(read['Leakages.momentum.p'][:k],axis=1)
    datas["E_dep"]=E_dep
    datas["Leak_sum"]=Leak_sum
    output.put((len(E_dep),datas,proc))
    output.put((-1,{},proc))
    return 0

def toh5single(sample):
    max_len=2e5
    reco_path="/pad/yulee/dream/repo/tools/reco"
    out_path=f"h5s/{sample}.h5py"
    print(out_path)
    hf=h5py.File(out_path,'w')
    count=0
    max_num=max(100,len(os.listdir(f"{reco_path}/{sample}")))
    compression='lzf'
    for j in tqdm.tqdm(range(max_num)):
        seed=j
        read_path=f"{reco_path}/{sample}/{sample}_{seed}.h5py"
        if(not os.path.isfile(read_path)):
            continue
        read=h5py.File(read_path,'r')
        k=0
        for i in range(len(read['GenParticles.PDG'])):
          gen_pdg=read['GenParticles.PDG'][i][0]
          if(gen_pdg==0):
              break
          k+=1
        if(count==0):
            for key in writekey:
              shape=list(read[key].shape)
              shape[0]=max_len
              hf.create_dataset(key,shape=shape,maxshape=shape,dtype=read[key].dtype,chunks=True,compression=compression)
            hf.create_dataset("E_dep",shape=[max_len],maxshape=[max_len],dtype="float64")
            hf.create_dataset("Leak_sum",shape=[max_len],maxshape=[max_len],dtype="float64")
        for key in writekey:
          hf[key][count:count+k]=read[key][:k]
          #gen_p=read['GenParticles.momentum.p'][i][0]
          #gen_phi=read['GenParticles.momentum.phi'][i][0]
          #gen_theta=read['GenParticles.momentum.theta'][i][0]
        E_dep=np.sum(read['SimCalorimeterHits.energy'][:k],axis=1)
        Leak_sum=np.sum(read['Leakages.momentum.p'][:k],axis=1)
        hf["E_dep"][count:count+k]=E_dep
        hf["Leak_sum"][count:count+k]=Leak_sum
        count+=k
        read.close()

    for key in hf.keys():
      hf[key].resize(count,axis=0)
      print(key,hf[key].shape)
    hf.close()

def read_dataset(file_name,dataset_name,k):
    with h5py.File(file_name,'r') as f:
        if(dataset_name=="E_dep"):
            return file_name,dataset_name,np.sum(f['SimCalorimeterHits.energy'][:k],axis=1)
        elif(dataset_name=="Leak_sum"):
            return file_name,dataset_name,np.sum(f['Leakages.momentum.p'][:k],axis=1)
        else:
            return file_name, dataset_name,f[dataset_name][:k]
def write_data(q,output_file,max_queue):
    with h5py.File(output_file,'a') as f:
        #print("write",q,output_file)
        for j in tqdm.tqdm(range(max_queue)):
            file_name,dataset_name,data_chunk=q.get()
            #print("chunk",file_name,dataset_name)
            if(file_name is None):
                break
            if(dataset_name in f):
                dataset = f[dataset_name]
                dataset.resize(dataset.shape[0]+data_chunk.shape[0],axis=0)
                #print(dataset_name,dataset.shape[0],data_chunk.shape[0])
                dataset[-data_chunk.shape[0]:]=data_chunk
            else:
                maxshape=(2e5,)+data_chunk.shape[1:]
                f.create_dataset(dataset_name,data=data_chunk,maxshape=maxshape,chunks=True,compression='lzf')
def merge_files(file_list,output_file,dataset_names=None):
    if(dataset_names is None):
        with h5py.File(file_list[0],'r') as f:
            dataset_names = list(f.keys())

    max_queue=int(len(file_list)*len(dataset_names)*1.05)
    q=multiprocessing.Queue()
    writer_process = multiprocessing.Process(target=write_data,args=(q,output_file,max_queue))
    writer_process.start()

    with multiprocessing.Pool(processes=8) as pool:
        results=[]
        for file_name in file_list:
            k=0
            with h5py.File(file_name,'r') as f:
                pdgs=f['GenParticles.PDG'][:]
                for i in range(len(pdgs)):
                  gen_pdg=pdgs[i][0]
                  if(gen_pdg==0):
                      break
                  k+=1
            if(k==0):continue
            for dataset_name in dataset_names:
                result = pool.apply_async(read_dataset,args=(file_name,dataset_name,k))
                results.append(result)
        for result in results:
            file_name,dataset_name,data_chunk=result.get()
            q.put((file_name,dataset_name,data_chunk))
    q.put((None,None,None))
    q.close()
    q.join_thread()
    writer_process.join()
def toh5(sample,max_num=1000):
    max_len=2e5
    reco_path="/pad/yulee/dream/repo/tools/reco"
    out_path=f"h5s/{sample}.h5py"
    hf=h5py.File(out_path,'w')
    hf.close()
    #max_num=max(100,len(os.listdir(f"{reco_path}/{sample}")))
    file_list=[]
    for seed in range(max_num):
        read_path=f"{reco_path}/{sample}/{sample}_{seed}.h5py"
        if(not os.path.isfile(read_path)):
            continue
        file_list.append(read_path)
    print(out_path,len(file_list))
    if(len(file_list)==0):print("Error: file_list is empty")
    merge_files(file_list,out_path,writekey)
    hf=h5py.File(out_path,'r')
    for key in writekey:
        #hf[key].resize(count,axis=0)
        print(key,hf[key].shape)
    hf.close()
pids= ['e-','pi+','pi0','gamma','kaon+','proton','neutron','kaon0L']
#pids= ['pi+','pi0','gamma','kaon+','proton','neutron','kaon0L']
for pid in pids:
  sample=f"{pid}_low"
  toh5(sample,1000)
  #sample=f"{pid}_0T"
  #toh5(sample,2500)
  #sample=f"{pid}_line"
  #toh5(sample,2500)
