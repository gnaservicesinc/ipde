from pathlib import Path
import numpy as np,cv2,json
from ipde.spatial import run_stereo_matching
p=Path('out/0758-investigation');l=np.load(p/'spatial_left.npy');r=np.load(p/'spatial_right.npy');sp=json.loads((p/'spatial.json').read_text());raft=np.load(p/'raft.npy')
sift=cv2.SIFT_create(nfeatures=8000);k1,d1=sift.detectAndCompute(cv2.cvtColor(l,cv2.COLOR_RGB2GRAY),None);k2,d2=sift.detectAndCompute(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),None)
good=[m for m,n in cv2.BFMatcher().knnMatch(d1,d2,k=2) if m.distance<.65*n.distance];pl=np.array([k1[m.queryIdx].pt for m in good]);pr=np.array([k2[m.trainIdx].pt for m in good]);dx=pl[:,0]-pr[:,0];dy=pl[:,1]-pr[:,1];keep=(dx>0)&(abs(dy)<20);pl=pl[keep];pr=pr[keep];A=np.column_stack((pr[:,0],pr[:,1],np.ones(len(pr))));delta=pl[:,1]-pr[:,1]
best=np.zeros(len(pr),bool);gen=np.random.default_rng(0)
for _ in range(1000):
 ind=gen.choice(len(pr),3,replace=False)
 try: fit=np.linalg.solve(A[ind],delta[ind])
 except np.linalg.LinAlgError: continue
 mask=abs(A@fit-delta)<1
 if mask.sum()>best.sum(): best=mask
fit=np.linalg.lstsq(A[best],delta[best],rcond=None)[0]; print('matches',len(pr),'inliers',best.sum(),'fit',fit,'residual',np.percentile(abs(A@fit-delta),[25,50,75,90,99]),'coverage',np.ptp(pr[best],axis=0))
for name,coef in [('shift',[0,0,np.median(delta[best])]),('affine',fit)]:
 M=np.array([[1,0,0],[coef[0],1+coef[1],coef[2]]]);aligned=cv2.warpAffine(r,M,(r.shape[1],r.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);np.save(p/f'right_{name}.npy',aligned)
 a=run_stereo_matching(l,aligned,sp).height_disparity_pixels;v=np.isfinite(a);err=abs(a[v]-raft[v]);np.save(p/f'aligned_{name}.npy',a);print(name,'valid',v.mean(),'err vs raft',np.percentile(err,[50,90,99]),'gt3',(err>3).mean())
np.savez(p/'registration_features.npz',left=pl,right=pr,inliers=best,fit=fit)
