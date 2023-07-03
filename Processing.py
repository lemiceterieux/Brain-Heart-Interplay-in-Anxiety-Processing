import numpy as np
import sys
import torch
from joblib import Parallel,delayed
import scipy.io as sio
from glob import glob
p = 8 
order = np.copy(p)
n_rois = 117
C = []
FullSpec = []
labs = []
trial = []
calls = ["RRI","PCG","MFG","ponsr","ponsd"]
HA = ["3a1","6a1","7a1","11a1","8a1","13a2","18a1","23a1","4a1","24a1","25a1","17a2","10a1","21a2"]
LA = ["3b2","6b1","7b1","11b1","8b1","13a1","18b2","23b1","4b1","24a2","25b1","17a1","10b2","21b1"]
for s in glob("s*/"): 
 ss = s[:3]
 z = np.unique(["_".join(s.split("_")[:3]) for s in glob("{0}/{0}_*.mat".format(ss))])
 labs += [zz.split("_")[-1] for zz in z]
 trial += [zz.split("_")[-2] for zz in z]
 print(s)#len(labs)//8)
 for zz in z:
  if sys.argv[-1] != "preload":
   band = [] 
   X = sio.loadmat(zz+"_bold.mat")
   XX = sio.loadmat(zz+"_rri.mat")
   T = np.min((XX['rri'].shape[1],X['bold'].shape[0]))
   tsplit = T//8
   for ttt in range(8):
     #  MVAR = np.concatenate((XX['rri'][:,:T],X['bold'][:T,[0,6,92,102]].T),0)
       MVAR = np.concatenate((XX['rri'][:,ttt*tsplit:ttt*tsplit+tsplit],X['bold'][ttt*tsplit:tsplit*ttt+tsplit,:].T),0)
       MVAR = MVAR.T
       MVAR = (MVAR - MVAR.mean(0))/MVAR.std(0)
       MVARp3 = MVAR[p:]
       MVARR = [MVAR[i:MVAR.shape[0]-p+i] for i in reversed(range(p))]
       MVARR += [np.ones_like(MVARR[-1][...,[0]])]
       MVARRc = np.concatenate(MVARR,-1)
       MVARRc = torch.from_numpy(MVARRc).cuda(0)
       MVARp3 = torch.from_numpy(MVARp3).cuda(0)
       W = torch.linalg.inv(MVARRc.T.matmul(MVARRc)).matmul(MVARRc.T).matmul(MVARp3).cpu().numpy()
       np.save("{1}_ARW.npy".format(ss,zz),W)
       freqs = 1.125/(2*tsplit)*np.arange(tsplit)
       fAR = np.zeros((n_rois,n_rois,tsplit),dtype=np.complex128)
       for kk in range(n_rois):
           for k in range(n_rois):
               delta = 0
               for i in range(p):
                   if kk == k:
                     delta = 1
                   fAR[k,kk] += delta-W[p*k+i,kk]*np.exp(-2*np.pi*freqs*i*1j)
       fAR = fAR.transpose(2,0,1)
       fARlow = fAR[np.logical_and(freqs > 0.05,freqs < 0.15)]
       fARhigh= fAR[np.logical_and(freqs > 0.1,freqs < 0.2)]
       fARhigher = fAR[np.logical_and(freqs > 0.2,freqs < 0.4)]
       Hlow = np.linalg.inv(fARlow)
       Hhigh = np.linalg.inv(fARhigh)
       Hhigher = np.linalg.inv(fARhigher)
       nHlow = np.zeros_like(Hlow,dtype=float)
       nHhigh = np.zeros_like(Hhigh,dtype=float)
       nHhigher = np.zeros_like(Hhigher,dtype=float)
       def makeLol(i):#for i in range(n_rois):
           nHigh = []
           nHigher = []
           nLow = []
           for j in range(n_rois):
               nHigh += [(abs(Hhigh[:,i,j])**2/np.sum(abs(Hhigh[:,i,:])**2,-1)).astype(float)]
               nHigher += [(abs(Hhigher[:,i,j])**2/np.sum(abs(Hhigher[:,i,:])**2,-1)).astype(float)]
               nLow += [(abs(Hlow[:,i,j])**2/np.sum(abs(Hlow[:,i,:])**2,-1)).astype(float)]
           return nHigher, nHigh, nLow
       nHhigher, nHhigh, nHlow = zip(*Parallel(n_jobs=117)(delayed(makeLol)(i) for i in range(n_rois)))
       band += [np.array([np.mean(nHlow,-1), np.mean(nHhigh,-1),np.mean(nHhigher,-1)])]
   bands = np.array(band).mean(0)
   np.save("{1}_AlldtfH.npy".format(ss,zz),bands)
  else:
   bands = np.load("{1}_AlldtfH.npy".format(ss,zz)).squeeze()
  C += [bands]
C = np.array(C)
trial = np.array(trial)
labs = np.array(labs)
A = C[trial=='a']
B = C[trial=='b']
C = (A + B)/2
rest1 = C[::4]
rest2 = C[1::4]
self = C[2::4]
visual = C[3::4]
Arest1 = A[::4]
Arest2 = A[1::4]
Aself = A[2::4]
Avisual = A[3::4]
Brest1 = B[::4]
Brest2 = B[1::4]
Bself = B[2::4]
Bvisual = B[3::4]

HAS = []
LAS = []
for i in range(14):
    print(HA[i],LA[i],HA[i][1],LA[i][1])
    if HA[i][1] == 'a':
        if HA[i][2] == '1':
            HAS += [Arest1[int(HA[i][0])]]
        else:
            HAS += [Arest2[int(HA[i][0])]]
    else:
        if HA[i][2] == '1':
            HAS += [Brest1[int(HA[i][0])]]
        else:
            HAS += [Brest2[int(HA[i][0])]]
    if LA[i][1] == 'a':
        if LA[i][2] == '1':
            LAS += [Arest1[int(LA[i][0])]]
        else:
            LAS += [Arest2[int(LA[i][0])]]
    else:
        if LA[i][2] == '1':
            LAS += [Brest1[int(LA[i][0])]]
        else:
            LAS += [Brest2[int(LA[i][0])]]
HA = np.array(HAS)
LA = np.array(LAS)

vr1 = []
vr2 = []
vs = []
vv = []
vvr1 = []
vvr2 = []
vvs = []
vvv = []

vh = []
vhh = []
vl = []
lm = (HA[:,0] - LA[:,0]).mean(0)
hm = (HA[:,1] - LA[:,1]).mean(0)
hhm = (HA[:,2] - LA[:,2]).mean(0)


for i in range(10000):
    p = np.random.permutation(HA.shape[0])
    p2 = np.random.permutation(LA.shape[0])
    vl += [np.mean(HA[p,0] - LA[p2,0] -0*lm,0)]
    vh += [np.mean(HA[p,1] - LA[p2,1] -0*hm,0)]
    vhh += [np.mean(HA[p,2] - LA[p2,2] -0*hm,0)]

tl = np.mean(HA[:,0] - LA[:,0],0)
th = np.mean(HA[:,1] - LA[:,1],0)
thh = np.mean(HA[:,2] - LA[:,2],0)
vh = np.array(vh)
vhh = np.array(vhh)
vl = np.array(vl)
pl = np.zeros((n_rois,n_rois),dtype=float)
ph = np.zeros((n_rois,n_rois),dtype=float)
phh = np.zeros((n_rois,n_rois),dtype=float)

for i in range(n_rois):
    for j in range(n_rois):
        pl[i,j] = np.min(((tl[i,j] > vl[:,i,j]).mean(0),(tl[i,j] < vl[:,i,j]).mean(0)))
        ph[i,j] = np.min(((th[i,j] > vh[:,i,j]).mean(0),(th[i,j] < vh[:,i,j]).mean(0)))
        phh[i,j] = np.min(((thh[i,j] > vhh[:,i,j]).mean(0),(thh[i,j] < vhh[:,i,j]).mean(0)))
print((pl<0.05)[:,0],"Low Band")
print((ph<0.05)[:,0],"High Band")

np.save("{0}_AllPVVs.npy".format(order),[ph,pl,phh])
np.save("{0}_AllmVVs.npy".format(order),[th,tl,thh])
