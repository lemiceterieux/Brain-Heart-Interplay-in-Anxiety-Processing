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
     #          nHhigh[:,i,j] = (abs(Hhigh[:,i,j])**2/np.sum(abs(Hhigh[:,i,:])**2,-1)).astype(float)
     #          nHlow[:,i,j] = (abs(Hlow[:,i,j])**2/np.sum(abs(Hlow[:,i,:])**2,-1)).astype(float)
       nHhigher, nHhigh, nHlow = zip(*Parallel(n_jobs=117)(delayed(makeLol)(i) for i in range(n_rois)))
     #  print(zz,np.shape(nHhigh))
       band += [np.array([np.mean(nHlow,-1), np.mean(nHhigh,-1),np.mean(nHhigher,-1)])]
   bands = np.array(band).mean(0)
   #np.save("{1}_AlldtfARW.npy".format(ss,zz),[fARlow,fARhigh])
   np.save("{1}_AlldtfH.npy".format(ss,zz),bands)
#print(W.shape,fAR.shape,zz,ss)
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

#r1m = (rest1[:14,0] - rest1[14:,0]).mean(0)
#r2m = (rest2[:14,0] - rest2[14:,0]).mean(0)
#vm = (visual[:14,0] - visual[14:,0]).mean(0)
#sm = (self[:14,0] - self[14:,0]).mean(0)

for i in range(10000):
    p = np.random.permutation(HA.shape[0])
    p2 = np.random.permutation(LA.shape[0])
    vl += [np.mean(HA[p,0] - LA[p2,0] -0*lm,0)]
    vh += [np.mean(HA[p,1] - LA[p2,1] -0*hm,0)]
    vhh += [np.mean(HA[p,2] - LA[p2,2] -0*hm,0)]
#    p = np.random.permutation(rest1.shape[0])  [:rest1.shape[0]//2]
#    pp = np.random.permutation(rest1.shape[0]) [:rest1.shape[0]//2]
#    p2 = np.random.permutation(rest1.shape[0]) [rest1.shape[0]//2:]
#    pp2 = np.random.permutation(rest1.shape[0])[rest1.shape[0]//2:]
#
#    vr1 += [np.mean(rest1[p,0] - rest1 [pp,0] -r1m,0)]
#    vr2 += [np.mean(rest2[p,0] - rest2 [pp,0] -r2m,0)]
#    vs +=  [np.mean(self[p,0] -  self  [pp,0] - sm,0)]
#    vv +=  [np.mean(visual[p,0]- visual[pp,0] - vm,0)]

tl = np.mean(HA[:,0] - LA[:,0],0)
th = np.mean(HA[:,1] - LA[:,1],0)
thh = np.mean(HA[:,2] - LA[:,2],0)
vh = np.array(vh)
vhh = np.array(vhh)
vl = np.array(vl)
#tr1 = np.mean(rest1[:14,0] - rest1[14:,0],0)
#tr2 = np.mean(rest2[:14,0] - rest2[14:,0],0)
#ts = np.mean(self[:14,0] - self[14:,0],0)
#tv = np.mean(visual[:14,0] - visual[14:,0],0)
#
#ttr1 = np.mean(rest1[:14,1] - rest1[14:,1],0)
#ttr2 = np.mean(rest2[:14,1] - rest2[14:,1],0)
#tts = np.mean(self[:14,1] - self[14:,1],0)
#ttv = np.mean(visual[:14,1] - visual[14:,1],0)
#
#vr1 = np.array(vr1)
#vr2 = np.array(vr2)
#vs = np.array(vs)
#vv = np.array(vv)
#
#vvr1 = np.array(vvr1)
#vvr2 = np.array(vvr2)
#vvs = np.array (vvs)
#vvv = np.array (vvv)

#pr1 = np.zeros((n_rois,n_rois),dtype=float)
#pr2 = np.zeros((n_rois,n_rois),dtype=float)
#ps = np.zeros((n_rois,n_rois),dtype=float)
#pv = np.zeros((n_rois,n_rois),dtype=float)
#ppr1 = np.zeros((n_rois,n_rois),dtype=float)
#ppr2 = np.zeros((n_rois,n_rois),dtype=float)
#pps = np.zeros((n_rois,n_rois),dtype=float)
#ppv = np.zeros((n_rois,n_rois),dtype=float)
pl = np.zeros((n_rois,n_rois),dtype=float)
ph = np.zeros((n_rois,n_rois),dtype=float)
phh = np.zeros((n_rois,n_rois),dtype=float)

for i in range(n_rois):
    for j in range(n_rois):
#        print((tr1[i,j] > vr1[:,i,j]).mean(0),(tr1[i,j] < vr1[:,i,j]).mean(0),calls[i],calls[j],"rest1")
#        print((tr2[i,j] > vr2[:,i,j]).mean(0),(tr2[i,j] < vr2[:,i,j]).mean(0),calls[i],calls[j],"rest2")
#        print((ts[i,j] > vs[:,i,j]).mean(0),(ts[i,j] < vs[:,i,j]).mean(0),calls[i],calls[j],"self")
#        print((tv[i,j] > vv[:,i,j]).mean(0),(tv[i,j] < vv[:,i,j]).mean(0),calls[i],calls[j],"visual",)
        pl[i,j] = np.min(((tl[i,j] > vl[:,i,j]).mean(0),(tl[i,j] < vl[:,i,j]).mean(0)))
        ph[i,j] = np.min(((th[i,j] > vh[:,i,j]).mean(0),(th[i,j] < vh[:,i,j]).mean(0)))
        phh[i,j] = np.min(((thh[i,j] > vhh[:,i,j]).mean(0),(thh[i,j] < vhh[:,i,j]).mean(0)))
#        pr1[i,j] = np.min(((tr1[i,j] > vr1[:,i,j]).mean(0),(tr1[i,j] < vr1[:,i,j]).mean(0)))
#        pr2[i,j] = np.min(((tr2[i,j] > vr2[:,i,j]).mean(0),(tr2[i,j] < vr2[:,i,j]).mean(0)))
#        pv[i,j] = np.min(((tv[i,j] > vv[:,i,j]).mean(0),(tv[i,j] < vv[:,i,j]).mean(0)))
#        ps[i,j] = np.min(((ts[i,j] > vs[:,i,j]).mean(0),(ts[i,j] < vs[:,i,j]).mean(0)))
#        ppr1[i,j] = np.min(((ttr1[i,j] > vvr1[:,i,j]).mean(0),(ttr1[i,j] < vvr1[:,i,j]).mean(0)))
#        ppr2[i,j] = np.min(((ttr2[i,j] > vvr2[:,i,j]).mean(0),(ttr2[i,j] < vvr2[:,i,j]).mean(0)))
#        ppv[i,j] = np.min((( ttv[i,j] >  vvv[:,i,j]).mean(0), (ttv[i,j] < vvv[:,i,j]).mean(0)))
#        pps[i,j] = np.min((( tts[i,j] >  vvs[:,i,j]).mean(0), (tts[i,j] < vvs[:,i,j]).mean(0)))
print((pl<0.05)[:,0],"Low Band")
print((ph<0.05)[:,0],"High Band")
#print((pr1<0.05)[:,0],"rest1")
#print((pr2<0.05)[:,0],"rest2")
#print((pv <0.05)[:,0],"vis")
#print((ps <0.05)[:,0],"self")
#print((ppr1<0.05)[:,0],"rest1")
#print((ppr2<0.05)[:,0],"rest2")
#print((ppv <0.05)[:,0],"vis")
#print((pps <0.05)[:,0],"self")
#print(tr1[:,0].astype(float),"rest1")
#print(tr2[:,0].astype(float),"rest2")
#print(tv [:,0].astype(float),"vis")
#print(ts [:,0].astype(float),"self")
#print(ttr1[:,0].astype(float),"rest1")
#print(ttr2[:,0].astype(float),"rest2")
#print(ttv [:,0].astype(float),"vis")
#print(tts [:,0].astype(float),"self")

np.save("{0}_AllPVVs.npy".format(order),[ph,pl,phh])#[pr1,pr2,ps,pv,ppr1,ppr2,pps,ppv])
np.save("{0}_AllmVVs.npy".format(order),[th,tl,thh])#[tr1,tr2,ts,tv,ttr1,ttr2,tts,ttv])
#print(rest1.shape,rest2.shape,self.shape,visual.shape,len(labs),np.unique(labs),np.unique(trial))
