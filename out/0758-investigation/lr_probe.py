import numpy as np,cv2,json
from pathlib import Path
from ipde.spatial import register_stereo_rows
p=Path('out/0758-investigation');l=np.load(p/'spatial_left.npy');r=np.load(p/'spatial_right.npy');r,valid,det=register_stereo_rows(l,r);print(det,flush=True)
matcher=cv2.StereoSGBM.create(minDisparity=0,numDisparities=336,blockSize=5,P1=600,P2=2400,disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=5,speckleWindowSize=50,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
f=matcher.compute(l,r).astype(np.float32)/16; rev=matcher.compute(np.ascontiguousarray(r[:,::-1]),np.ascontiguousarray(l[:,::-1]))[:,::-1].astype(np.float32)/16
xr=np.arange(f.shape[1])[None,:]-f;xi=np.clip(np.round(xr),0,f.shape[1]-1).astype(int); dr=rev[np.arange(f.shape[0])[:,None],xi];v=(f>=0)&(dr>=0)&(abs(f-dr)<=1)&(xr>=0)&(xr<=f.shape[1]-1);f[~v]=np.nan;np.save(p/'lr_checked.npy',f)
raft=np.load(p/'raft.npy');err=abs(f[v]-raft[v]);print('valid',v.mean(),'err',np.percentile(err,[50,90,99]),'gt3',(err>3).mean(),'disparity',np.percentile(f[v],[0,1,10,50,90,99,100]),flush=True)
