import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from nilearn import datasets
import sys
if sys.argv[-2] != "assess.py":
    order = sys.argv[-2]
else:
    order = ""
aal = datasets.fetch_atlas_aal()
psort = np.sort(np.load(order + "AllPVVs.npy"))
psort.shape
psort = np.sort(np.load(order + "AllPVVs.npy").ravel())
psort.shape
below = psort < 0.01*np.arange(len(psort))/len(psort)

print(below.sum()/len(psort)*100,len(psort))
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[0,0,:])[0]-1])
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[1,0,:])[0]-1])
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[2,0,:])[0]-1])
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[0,:,0])[0]-1])
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[1,:,0])[0]-1])
print(np.array(aal.labels)[np.where((np.load(order + "AllPVVs.npy") < psort[np.where(below)[0].max()])[2,:,0])[0]-1])
labels = ["RR_Interval"] + aal.labels
data = np.load(order + "AllPVVs.npy")
dataT = np.load(order + "AllmVVs.npy")

print(len(labels),data.shape)
print()
print()
hemi = "Right"

for hemi in ["Right","Left",""]:
    if sys.argv[-1] == "FDR":
        thresh = psort[np.where(below)[0].max()]
        prefix = order + "_FDR_"
    else:
        thresh = 0.01
        prefix = order + hemi
    print(thresh,psort[np.where(below)[0].max()])
    if hemi == "Right":
        labs = [0] + (np.arange(0,116,2)+1).tolist()
    elif hemi == "Left":
        labs = [0] + (np.arange(0,116,2)+2).tolist()
    else:
        labs = [0] + (np.arange(0,116)).tolist()
    
    print(data.shape)
    with open(prefix + "LowBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
        plt.imshow(data[0][labs][:,labs]<psort[np.where(below)[0].max()],cmap='gray')
        plt.colorbar()
        plt.savefig(prefix+"LowBand.png")
        plt.close('all')
    
        for i in labs:
            for j in labs:
                if data[0,i,j] < thresh:#psort[np.where(below)[0].max()]:
                    L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[0,i,j],i,j))
                    if data[0,i,j] < psort[np.where(below)[0].max()]:
                        L.write("\tTrue")
                    else:
                        L.write("\tFalse")
                    L.write("\n")
    with open(prefix + "RRLowBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
    
        for i in labs:
            for j in labs:
                if i == 0 or j == 0:
                    if data[0,i,j] < thresh:# psort[np.where(below)[0].max()]:
                        L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[0,i,j],i,j))
                        if data[0,i,j] < psort[np.where(below)[0].max()]:
                            L.write("\tTrue")
                        else:
                            L.write("\tFalse")
                        L.write("\n")
    
    with open(prefix + "HighBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
        plt.imshow(data[1][labs][:,labs]<psort[np.where(below)[0].max()],cmap='gray')
        plt.colorbar()
        plt.savefig(prefix+"HighBand.png")
        plt.close('all')
    
        for i in labs:
            for j in labs:
                if data[1,i,j] < thresh:# psort[np.where(below)[0].max()]:
                    L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[1,i,j],i,j))
                    if data[1,i,j] < psort[np.where(below)[0].max()]:
                        L.write("\tTrue")
                    else:
                        L.write("\tFalse")
                    L.write("\n")
      
    with open(prefix + "RRHighBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
    
        for i in labs:
            for j in labs:
                if i == 0 or j == 0:
                    if data[1,i,j] < thresh:# psort[np.where(below)[0].max()]:
                        L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[1,i,j],i,j))
                        if data[1,i,j] < psort[np.where(below)[0].max()]:
                            L.write("\tTrue")
                        else:
                            L.write("\tFalse")
                        L.write("\n")
    
    with open(prefix + "HigherBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
        plt.imshow(data[2][labs][:,labs]<psort[np.where(below)[0].max()],cmap='gray')
        plt.colorbar()
        plt.savefig(prefix + "HigherBand.png")
        plt.close('all')
    
        for i in labs:
            for j in labs:
                if data[2,i,j] < thresh:# psort[np.where(below)[0].max()]:
                    L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[2,i,j],i,j))
                    if data[2,i,j] < psort[np.where(below)[0].max()]:
                        L.write("\tTrue")
                    else:
                        L.write("\tFalse")
                    L.write("\n")
      
    with open(prefix + "RRHigherBandConnections.txt",'w') as L:
        L.write("Connection \tDTF Output\tFDR\n")
        L.write("________________________________\n")
    
        for i in labs:
            for j in labs:
                if i == 0 or j == 0:
                    if data[2,i,j] < thresh:# psort[np.where(below)[0].max()]:
                        L.write("{3} {0}-->{4} {1} \t{2:.3e}".format(labels[i],labels[j],dataT[2,i,j],i,j))
                        if data[2,i,j] < psort[np.where(below)[0].max()]:
                            L.write("\tTrue")
                        else:
                            L.write("\tFalse")
                        L.write("\n")
    
