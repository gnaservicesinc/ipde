from pathlib import Path
import cv2,numpy as np,json,time
from ipde.spatial import run_stereo_matching,StereoMatchingOptions,histogram_match_stereo_pair
p=Path('out/0758-investigation');l=np.load(p/'spatial_left.npy');r=np.load(p/'spatial_right.npy');raft=np.load(p/'raft.npy');sp=json.loads((p/'spatial.json').read_text())
sift=cv2.SIFT_create(nfeatures=4000)
k1,d1=sift.detectAndCompute(cv2.cvtColor(l,cv2.COLOR_RGB2GRAY),None);k2,d2=sift.detectAndCompute(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),None)
good=[m for m,n in cv2.BFMatcher().knnMatch(d1,d2,k=2) if m.distance<.65*n.distance]
pts1=np.array([k1[m.queryIdx].pt for m in good]);pts2=np.array([k2[m.trainIdx].pt for m in good]);diff=pts1-pts2
valid=(diff[:,0]>0)&(abs(diff[:,1])<5);dy=diff[valid,1];print('features',len(good),'epipolar',valid.sum(),'dy percentiles',np.percentile(dy,[1,25,50,75,99]),flush=True)
for name,left,right,block in [('raw5',l,r,5),('raw11',l,r,11),('gray5',np.repeat(cv2.cvtColor(l,cv2.COLOR_RGB2GRAY)[:,:,None],3,2),np.repeat(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)[:,:,None],3,2),5),('matched5',*[(lambda v:v)(v) for v in (histogram_match_stereo_pair(l,r).left,histogram_match_stereo_pair(l,r).right)],5)]:
 t=time.monotonic();result=run_stereo_matching(left,right,sp,StereoMatchingOptions(block_size=block));a=result.height_disparity_pixels;v=np.isfinite(a);err=abs(a[v]-raft[v]);np.save(p/f'{name}.npy',a)
 print(name,'seconds',time.monotonic()-t,'valid',v.mean(),'err vs raft',np.percentile(err,[50,90,99]),'gt3',(err>3).mean(),flush=True)
