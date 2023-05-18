from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets,plotting,surface

aal2 = datasets.fetch_atlas_aal()
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
aal = datasets.fetch_atlas_aal()
hemi = 'left'
radius = 0.
pial_mesh = fsaverage['pial_' + hemi]
lh = surface.vol_to_surf(aal.maps, pial_mesh, radius=radius)

hemi = 'right'
radius = 0.
pial_mesh = fsaverage['pial_' + hemi]
rh = surface.vol_to_surf(aal.maps, pial_mesh, radius=radius)

ToPlot = [s for s in glob("*.txt")]
ToPlot = ["Left0204.txt"
,"Right0204.txt"
,"Left0102.txt"
,"Right0102.txt"
,"Left005015.txt","Right005015.txt"
]
Files = ["Left Hemisphere, 0.2 Hz to 0.4 Hz", "Right Hemisphere, 0.2 Hz to 0.4 Hz"]
Files += ["Left Hemisphere, 0.1 Hz to 0.2 Hz", "Right Hemisphere, 0.1 Hz to 0.2 Hz"]
Files += ["Left Hemisphere, 0.05 Hz to 0.15 Hz", "Right Hemisphere, 0.05 Hz to 0.15 Hz"]
Ascendingdata = [[] for i in range(len(ToPlot))]
Ascendinglabs = [[] for i in range(len(ToPlot))]
Ascendinginds = [[] for i in range(len(ToPlot))]
Descendingdata = [[] for i in range(len(ToPlot))]
Descendinglabs = [[] for i in range(len(ToPlot))]
Descendinginds = [[] for i in range(len(ToPlot))]
for i,s in enumerate(ToPlot):
    if "Left" in s:
        app = "_L"
    else:
        app = "_R"
    with open(s,'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            temp = l.split("\t")
            if "Vermis" in temp[0]:
                print(temp)
            if "Interval-->" in temp[0]:
                temppp = temp[0].split("-->")
                if "Interval" in temppp[1]:
                    continue
                Ascendinglabs[i].append("_".join(temppp[1].split("_")[:-1]))
                Ascendingdata[i].append(float(temp[1]))
                for kkk in range(len(aal.labels)):
                    trying = Ascendinglabs[i][-1]
                    if trying+app in aal.labels[kkk]:
                        Ascendinginds[i].append(float(aal.indices[kkk]))
                        break
            else:
                temppp = temp[0].split("-->")
                Descendinglabs[i].append("_".join(temppp[0].split("_")[:-1]))
                Descendingdata[i].append(float(temp[1]))
                for kkk in range(len(aal.labels)):
                    trying = Descendinglabs[i][-1]
                    if trying+app in aal.labels[kkk]:
                       Descendinginds[i].append(float(aal.indices[kkk]))
                       break
print(Descendinginds,Ascendinginds)
      
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams.update({'font.size': 10})

for Al, Ad,F,T in zip(Ascendinglabs,Ascendingdata,Files,ToPlot):
	fname = T.split(".")[0] + ".png"
	if len(Al) == 0:
		continue
	plt.bar(np.arange(len(Al)),Ad)
	plt.xticks(np.arange(len(Al)),[l for l in Al],rotation = 85)
	plt.tight_layout()
	plt.savefig("Ascending_" + fname)
	plt.close('all')

for Al, Ad,F,T in zip(Descendinglabs,Descendingdata,Files,ToPlot):
	fname = T.split(".")[0] + ".png"
	if len(Al) == 0:
		continue
	plt.bar(np.arange(len(Al)),Ad)
	plt.xticks(np.arange(len(Al)),[l for l in Al],rotation = 85)
	plt.tight_layout()
	plt.savefig("Descending_" + fname)
	plt.close('all')

fig,ax = plt.subplots(3,2,sharey=True,figsize=(12,12))
for Al, Ad,F,T,a in zip(Ascendinglabs,Ascendingdata,Files,ToPlot,ax.ravel()):
	fname = T.split(".")[0] + ".png"
	if len(Al) == 0:
		a.set_xticks([],[])
		continue
	a.bar(np.arange(len(Al)),Ad)
#	a.set_title(F)
	a.set_xticks(np.arange(len(Al)),[l for l in Al],rotation = 85)
plt.tight_layout()
plt.savefig("Ascending",bbox_inches='tight')
plt.close('all')

fig,ax = plt.subplots(3,2,sharey=True,figsize=(12,12))
for Al, Ad,F,T,a in zip(Descendinglabs,Descendingdata,Files,ToPlot,ax.ravel()):
	fname = T.split(".")[0] + ".png"
	if len(Al) == 0:
		a.set_xticks([],[])
		continue
	a.bar(np.arange(len(Al)),Ad)
#	a.set_title(F)
	a.set_xticks(np.arange(len(Al)),[l for l in Al],rotation = 85)
plt.tight_layout()
plt.savefig("Descending",bbox_inches='tight')
plt.close('all')


lasctexture02 = np.zeros(len(lh))
rasctexture02 = np.zeros(len(lh))
lasctexture01 = np.zeros(len(lh))
rasctexture01 = np.zeros(len(lh))
lasctexture005 = np.zeros(len(lh))
rasctexture005 = np.zeros(len(lh))

ldesctexture02 = np.zeros(len(lh))
rdesctexture02 = np.zeros(len(lh))
ldesctexture01 = np.zeros(len(lh))
rdesctexture01 = np.zeros(len(lh))
ldesctexture005 = np.zeros(len(lh))
rdesctexture005 = np.zeros(len(lh))

for Al, Ad,F,T,a in zip(Ascendinglabs,Ascendingdata,Files,ToPlot,Ascendinginds):
    fname = T.split(".")[0] + ".png"
    if len(Al) == 0:
        continue
    for aa, ad in zip(a,Ad):
        if "Left" in T:
            if "0204" in T:
                lasctexture02[lh==aa] = ad
            elif "0102" in T:
                lasctexture01[lh==aa] = ad
            elif "005015" in T:
                lasctexture005[lh==aa] = ad
        else:
            if "0204" in T:
                rasctexture02[rh==aa] = ad
            elif "0102" in T:
                rasctexture01[rh==aa] = ad
            elif "005015" in T:
                rasctexture005[rh==aa] = ad
parcel = rh
rhands = Ascendinglabs[1]
regions_indices = Ascendinginds[1]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_0204_Asc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_0204_Asc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_0204_Asc.png",bbox_inches='tight')
plt.close('all')

parcel = rh
rhands = Ascendinglabs[3]
regions_indices = Ascendinginds[3]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_0101_Asc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_0102_Asc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_0102_Asc.png",bbox_inches='tight')
plt.close('all')

parcel = rh
rhands = Ascendinglabs[5]
regions_indices = Ascendinginds[5]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rasctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_005015_Asc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_005015_Asc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_005015_Asc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[1]
regions_indices = Ascendinginds[1]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_0204_Asc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_0204_Asc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_0204_Asc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[3]
regions_indices = Ascendinginds[3]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_0101_Asc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_0102_Asc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_0102_Asc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[5]
regions_indices = Ascendinginds[5]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, lasctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_005015_Asc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_005015_Asc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_005015_Asc.png",bbox_inches='tight')
plt.close('all')

for Al, Ad,F,T,a in zip(Descendinglabs,Descendingdata,Files,ToPlot,Descendinginds):
    fname = T.split(".")[0] + ".png"
    if len(Al) == 0:
        continue
    for aa, ad in zip(a,Ad):
        if "Left" in T:
            if "0204" in T:
                ldesctexture02[lh==aa] = ad
            elif "0102" in T:
                ldesctexture01[lh==aa] = ad
            elif "005015" in T:
                ldesctexture005[lh==aa] = ad
        else:
            if "0204" in T:
                rdesctexture02[rh==aa] = ad
            elif "0102" in T:
                rdesctexture01[rh==aa] = ad
            elif "005015" in T:
                rdesctexture005[rh==aa] = ad


parcel = rh
rhands = Ascendinglabs[1]
regions_indices = Ascendinginds[1]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture02, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_0204_Desc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_0204_Desc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_0204_Desc.png",bbox_inches='tight')
plt.close('all')

parcel = rh
rhands = Ascendinglabs[3]
regions_indices = Ascendinginds[3]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture01, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_0101_Desc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_0102_Desc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_0102_Desc.png",bbox_inches='tight')
plt.close('all')

parcel = rh
rhands = Ascendinglabs[5]
regions_indices = Ascendinginds[5]

fig = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_right, rdesctexture005, cmap='seismic',hemi='right',threshold=1e-4, bg_map=fsaverage.sulc_right,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_right, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("RResSurfLat_005015_Desc.png",bbox_inches='tight')
fig2.savefig("RResSurfMed_005015_Desc.png",bbox_inches='tight')
fig3.savefig("RResSurfVent_005015_Desc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[1]
regions_indices = Ascendinginds[1]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture02, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_0204_Desc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_0204_Desc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_0204_Desc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[3]
regions_indices = Ascendinginds[3]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture01, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=lhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_0101_Desc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_0102_Desc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_0102_Desc.png",bbox_inches='tight')
plt.close('all')

parcel = lh
lhands = Ascendinglabs[5]
regions_indices = Ascendinginds[5]

fig = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig, legend=False)
fig2 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='medial',vmax=0.1,colorbar=None)
#plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig2, legend=False)
fig3 = plotting.plot_surf_stat_map(fsaverage.infl_left, ldesctexture005, cmap='seismic',hemi='left',threshold=1e-4, bg_map=fsaverage.sulc_left,view='ventral',vmax=0.1,colorbar=True)
#fig3 = plotting.plot_surf_contours(fsaverage.infl_left, parcel, labels=rhands, levels=regions_indices, figure=fig3, legend=False)
fig.savefig("LResSurfLat_005015_Desc.png",bbox_inches='tight')
fig2.savefig("LResSurfMed_005015_Desc.png",bbox_inches='tight')
fig3.savefig("LResSurfVent_005015_Desc.png",bbox_inches='tight')
plt.close('all')
