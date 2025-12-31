# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:34:17 2024

@author: Jong Hoon Lee
"""

# import packages 

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA

from os.path import join as pjoin
from numba import jit, cuda

from rastermap import Rastermap, utils
from scipy.stats import zscore

# %% File name and directory

fname = 'CaData_all_session_v3_corrected.mat'
fdir = 'D:\Python\Data'


# %% Helper functions for loading and selecting data
# 


np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 


def find_good_data():
    D_ppc = load_matfile()
    good_list = []
    for n in range(np.size(D_ppc,0)):
        S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
        for sp in np.array(D_ppc[n,0]):
            if sp < np.size(S_all,1):
                S_all[0,sp[0]-1] = 1  #spike time starts at 1 but indexing starts at 0
                
        if np.mean(S_all)*1e3>1:
            good_list = np.concatenate((good_list,[n]))
    return good_list


def find_good_data_Ca(t_period):
    D_ppc = load_matfile_Ca()
    good_list = []
    t_period = t_period+prestim

    for n in range(np.size(D_ppc,0)):
        N_trial = np.size(D_ppc[n,2],0)
    
    
    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y) > 0.5:
            good_list = np.concatenate((good_list,[n]))
    
    
    return good_list
alph = 0.8
def randorg2(listd):
    t_list = np.where(listd == True)
    listd2 = listd[:]
    listd2[:] = False
    listd2[t_list[0][:int(np.floor(np.size(t_list[0])*alph))]] = True
    return listd2

def find_good_data_Ca2(t_period):
    D_ppc = load_matfile_Ca()
    good_list = []
    t_period = t_period+prestim

    for n in range(np.size(D_ppc,0)):
        N_trial = np.size(D_ppc[n,2],0)
    
        ttr = D_ppc[n,4][0][0]

    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y[:200,:]) > 0.5 :
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[200:ttr+26,:]) > 0.5:
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[ttr+26:N_trial,:])> 0.5 :
            good_list = np.concatenate((good_list,[n]))
    
    return good_list



@jit(target_backend='cuda')                         
def import_data_w_spikes(n,prestim,t_period,window,c_ind):
    D_ppc = load_matfile()
    S_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    L_all = np.zeros((1,max(D_ppc[n,2][:,0])+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting spikes from data
    for sp in np.array(D_ppc[n,0]):
        if sp < np.size(S_all,1):
            S_all[0,sp[0]-1] = 1  #spike time starts at 1 but indexing starts at 0
                
    
    S = np.zeros((N_trial,t_period))
    S_pre = np.zeros((N_trial,prestim))
    for tr in range(N_trial):
        S[tr,:] = S_all[0,D_ppc[n,2][tr,0]-1:D_ppc[n,2][tr,0]+t_period-1]
        S_pre[tr,:] = S_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]-1]
    
    # extracting spikes, end
    
    # extracting licks, the same way
    # for l in np.array(D_ppc[n,1]):
    #     if l < np.size(L_all,1):
    #         L_all[0,l[0]-1] = 1 
    
    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        L[tr,:] = L_all[0,D_ppc[n,2][tr,0]-prestim-1:D_ppc[n,2][tr,0]+t_period-1]
    
    
    X = D_ppc[n,2][:,2:6] # task variables
    Y = [];
    Y2 = [];
    S = np.concatenate((S_pre,S),1)
    t_period = t_period+prestim
    
    
    if c_ind !=3:
    # remove conditioning trials     
        S = np.concatenate((S[0:200,:],S[D_ppc[n,5][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,5][0][0]:,:]),0)
        L = np.concatenate((L[0:200,:],L[D_ppc[n,5][0][0]:,:]),0)

    # only contain conditioningt trials
    else:
        S = S[201:D_ppc[n,5][0][0]]
        X = X[201:D_ppc[n,5][0][0]]


    N_trial2 = np.size(S,0)

    # select analysis and model parameters with c_ind    
    
    if c_ind == -1:                
        # Adding previous trial correct vs wrong
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]
        X = np.concatenate((X,Xpre),1)
        X2 = X
    elif c_ind ==-2:
        Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
        Xpre = Xpre[:,None]       
        X2 = np.column_stack([X[:,0],X[:,3],
                             X[:,2]*X[:,1],Xpre])  
    
    
    
    L2 = []
    for w in range(int(t_period/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l))
        y = np.mean(S[:,range(window*w,window*(w+1))],1)*1e3
        y2 = np.sum(S[:,range(window*w,window*(w+1))],1)
        Y = np.concatenate((Y,y))
        Y2 = np.concatenate((Y2,y2))
        
    Y = np.reshape(Y,(int(t_period/window),N_trial2)).T
    Y2 = np.reshape(Y2,(int(t_period/window),N_trial2)).T
    L2 = np.reshape(L2,(int(t_period/window),N_trial2)).T
    return X2, Y, Y2, L2

def import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr):    
    
    L_all = np.zeros((1,int(np.floor(D_ppc[n,3][0,max(D_ppc[n,2][:,0])]*1e3))+t_period+100))
    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    # for l in np.floor(D_ppc[n,1]*1e3):
    #     # l = int(l) 
    #     if int(l) < np.size(L_all,1):
    #         L_all[0,l-1] = 1 
    
    L = np.zeros((N_trial,t_period+prestim))
    for tr in range(N_trial):
        stim_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,0]]*1e3))
        lick_onset = int(np.round(D_ppc[n,3][0,D_ppc[n,2][tr,3]]*1e3))
        lick_onset = lick_onset-stim_onset
        L[tr,:] = L_all[0,stim_onset-prestim-1:stim_onset+t_period-1]
        
        # reformatting lick rates
    L2 = []
    for w in range(int((t_period+prestim)/window)):
        l = np.sum(L[:,range(window*w,window*(w+1))],1)
        L2 = np.concatenate((L2,l)) 
            
    L2 = np.reshape(L2,(int((t_period+prestim)/window),N_trial)).T


    X = D_ppc[n,2][:,2:6] # task variables
    Rt =  D_ppc[n,5] # reward time relative to stim onset, in seconds
    t_period = t_period+prestim
    
    # re-formatting Ca traces
    Yraw = {}
    Yraw = D_ppc[n,0]
    
    # Original Y calculation #####
    
    Y = np.zeros((N_trial,int(t_period/window)))
    for tr in range(N_trial):
        Y[tr,:] = Yraw[0,D_ppc[n,2][tr,0]-1 - int(prestim/window): D_ppc[n,2][tr,0] + int(t_period/window)-1 - int(prestim/window)]

                
    # select analysis and model parameters with c_ind
    
    if c_ind ==0:             
    # remove conditioning trials 
        Y = np.concatenate((Y[0:200,:],Y[D_ppc[n,4][0][0]:,:]),0)
        X = np.concatenate((X[0:200,:],X[D_ppc[n,4][0][0]:,:]),0)
        L2 = np.concatenate((L2[0:200,:],L2[D_ppc[n,4][0][0]:,:]),0)
    elif c_ind == 1:        
        Y = Y[0:200,:]
        X = X[0:200,:]
        L2 = L2[0:200,:]
    elif c_ind ==2:
        Y = Y[D_ppc[n,4][0][0]:,:]
        X = X[D_ppc[n,4][0][0]:,:]
        L2 = L2[D_ppc[n,4][0][0]:,:]
    elif c_ind ==3:
        if ttr == -1:
            c1 = 200 
            c2 = D_ppc[n,4][0][0]+ 25
        elif ttr < 4:            
            c1 = ttr*50
            c2 = c1 +50
        else:
            c1 = D_ppc[n,4][0][0]+(ttr-4)*50 + 25
            c2 = c1+ 50
            if ttr == 7:
                c2 = np.size(X,0)
            else:
                c2 = c1+50

        
        Y = Y[c1:c2]
        X = X[c1:c2]
        L2 = L2[c1:c2]

    
    # Add reward  history
    Xpre = np.concatenate(([0],X[0:-1,2]*X[0:-1,1]),0)
    Xpre = Xpre[:,None]
    Xpre2 = np.concatenate(([0,0],X[0:-2,2]*X[0:-2,1]),0)
    Xpre2 = Xpre2[:,None]
    # Add reward instead of action
    X2 = np.column_stack([X[:,0],X[:,3],
                          X[:,2]*X[:,1],Xpre]) 

    

    
    return X,Y, L2, Rt


# %%

t_period = 6000
prestim = 2000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
#k = 10 # number of cv
ca = 1

if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca2(t_period)
    
# %% get neural data
# Note that Y is the neural data segmented by trial.
# X is the trial type information, which allows us to select specific trials for Y
# first row is Go/Nogo. Second row is Lick/No-lick
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""

lenx = 160 # Length of data, 8000ms, with a 50 ms window.
D_all = np.zeros((len(good_list),lenx))
D = {}
trmax = 8
alpha = 1
for tr in np.arange(trmax):
    D[0,tr] = np.zeros((len(good_list),lenx))
    D[1,tr] = np.zeros((len(good_list),lenx))
    D[2,tr] = np.zeros((len(good_list),lenx))



for ind in [0,1,2]:
    D[ind,trmax] = np.zeros((len(good_list),lenx))
    D[ind,trmax+1] = np.zeros((len(good_list),lenx))

c_ind = 3
Y = {}
for tr in np.arange(trmax):
    print(tr)
    m = 0
    ttr = tr
    if tr == 4:
        ttr = -1
    elif tr >4:
        ttr = tr-1
        
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        for ind in [0,1,2]:
            if ind == 0:
                # CR(correct): contingency==0 & lick==0 & correct==1
               # D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 0), :], 0)
                D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 0)*(X[:,1] == 0),:],0) # CR
            elif ind == 2:
                # FA : contingency==0 & lick==1
              #  D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 0), :], 0)
              D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 0)*(X[:,1] == 1),:],0) # FA
            else:
                # D[ind,trmax+c_ind-1][m,:] = np.mean(Y[X[:,0] == ind,:],0) # GO 
                # Hit: contingency==1 & lick==1 & correct==1
              #  D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 1), :], 0)
                D[ind,tr][m,:] = np.mean(Y[(X[:,0] == 1)*(X[:,2] == 1),:],0) #Hit

            max1 = np.nanmax(np.mean(Y[(X[:,0] == 1)*(X[:,1] == 1)*(X[:,2] == 1), :], 0))  # Hit
            max2 = np.nanmax(np.mean(Y[(X[:,0] == 0)*(X[:,1] == 0)*(X[:,2] == 1), :], 0))  # CR(correct)
            D[ind,tr][m,:] = D[ind,tr][m,:]/(np.nanmax([max1,max2])  + alpha) # for original trajectories with Go+ NG
        m += 1
        
for c_ind in[1,2]:
    m = 0
    for n in good_list: 
        n = int(n)
        X, Y, L, Rt = import_data_w_Ca(D_ppc,n,prestim,t_period,window,c_ind,ttr)
        for ind in [0,1,2]:
            if ind == 0:
                    D[ind, trmax + c_ind - 1][m, :] = np.mean(Y[(X[:,0] == 0)*(X[:,1] == 0)*(X[:,2] == 1), :], 0)
            elif ind == 2:
                    D[ind, trmax + c_ind - 1][m, :] = np.mean(Y[(X[:,0] == 0)*(X[:,1] == 1), :], 0)
            else:
                    D[ind, trmax + c_ind - 1][m, :] = np.mean(Y[(X[:,0] == 1)*(X[:,1] == 1)*(X[:,2] == 1), :], 0)
            max1 = np.nanmax(np.mean(Y[(X[:,0] == 1)*(X[:,1] == 1)*(X[:,2] == 1), :], 0))
            max2 = np.nanmax(np.mean(Y[(X[:,0] == 0)*(X[:,1] == 0)*(X[:,2] == 1), :], 0))
            D[ind,trmax+c_ind-1][m,:] = D[ind,trmax+c_ind-1][m,:]/(np.max([max1,max2]) + alpha)
        m += 1
    

# %% sort units by peak order

d_list = good_list > 194 #PPCAC
d_list3 = good_list <= 194 # PPCIC


d_list2 = d_list # select here for PPCIC or PPCAC
D_r = {}
sm = 2 # smoothing parameter rho
g_ind = 1 # group index, this is where data is sorted to 
trmax = 8 # max number of sections


for g in np.arange(trmax+2):
    D_r[g] = np.concatenate((D[1,g][d_list2,:],D[0,g][d_list2,:]),1)
    D_r[g] = ndimage.gaussian_filter(D_r[g],[0,sm])



max_ind = np.argmax(D_r[0][:,:],1)
max_peaks = np.argsort(max_ind)
clim = [-1,1.5]



fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))
for ind, ax in zip(np.arange(8), axs.ravel()):
    ax.imshow(zscore(D_r[ind][max_peaks, :],axis = 1),clim = clim, cmap="viridis", vmin=clim[0], vmax=clim[1], aspect="auto")


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(40, 15))
im1 =axs[0].imshow(zscore(D_r[8][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
im1 =axs[1].imshow(zscore(D_r[9][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
im1 =axs[2].imshow(zscore(D_r[4][max_peaks, :],axis = 1),clim = clim, cmap="viridis",  vmin=clim[0], vmax=clim[1], aspect="auto")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

# %% rastermap
# sorting using rastermap. Figure 4c and 4d

spks = np.concatenate((D_r[8],D_r[4],D_r[9]),1)

model = Rastermap(n_PCs =50,
                  locality=0.5,
                  time_lag_window=10,
                  n_clusters =50,
                  grid_upsample=2, keep_norm_X = False).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)
clim = [-.5, 1.5]
    
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

im2 = axs[0].imshow(zscore((D_r[8][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
im2 = axs[1].imshow(zscore((D_r[4][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
im2 = axs[2].imshow(zscore((D_r[9][isort, :]),axis = 1),clim = clim, cmap="gray_r",  vmin=clim[0], vmax=clim[1], aspect="auto")
v = clim

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)



# %% PCA on individual groups for subspace overlap calculation

def listshuffle(p_list, fract): # use this function to run subspace analysis n-fold
    for n in np.arange(len(p_list)):
        if p_list[n] == True:
            p_list[n] = np.random.choice([True, False],p = [fract, 1-fract])
    return p_list
pca = {}
max_k = 20;
n_cv = 20   
trmax = 8
d_list2 = d_list
# fig, axs = plt.subplots(trmax+2,6,figsize = (20,30))

sm = 0
R = {}
test= {}
for k in np.arange(n_cv):
    d_list3 = good_list > 194 # change here for PPCIC or PPCAC
    d_list2 = listshuffle(randorg2(d_list3[:]),0.8)
    for g in  np.arange(trmax+2):
        pca[k,g] = PCA(n_components=50)
        R[g] = np.concatenate((D[1,g][d_list2,:].T,D[0,g][d_list2,:].T),0)
        test = pca[k,g].fit_transform(ndimage.gaussian_filter(R[g],[5,0]))        
        test = test.T
# %% subspace overlap calculation
# for explanation of method, see Bondanelli et al., 2021
from scipy import linalg

n_cv = 1 # you may increase the cross validation number here    
trmax = 8
Overlap = np.zeros((trmax-1,trmax-1,n_cv));


k1 = 20 #20
k2 = 19 #19

U = {}
for k in np.arange(n_cv):
    for g1 in [1,2,3,4,5,6,7]: #np.arange(trmax):
       for g2 in [1,2,3,4,5,6,7]: # np.arange(trmax):
           S_value = np.zeros((1,k1))
           for d in np.arange(0,k1):
               S_value[0,d] = np.abs(np.dot(pca[k,g1].components_[d,:], pca[k,g2].components_[d,:].T))
               S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[k,g1].components_[d,:])*np.linalg.norm(pca[k,g2].components_[d,:]))
           Overlap[g1-1,g2-1,k] = np.max(S_value)
fig, axes = plt.subplots(1,1,figsize = (10,10))

        
imshowobj = axes.imshow(np.mean(Overlap[:,:,:],2),cmap = "hot_r")
imshowobj.set_clim(0.3, 0.9) #correct
plt.colorbar(imshowobj) #adjusts scale to value range

