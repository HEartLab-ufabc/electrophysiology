#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:38:48 2021

@author: italo
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as sig
# import scipy.io as sio
# import os

def sawtooth(x, order):
    vec = np.zeros_like(x)
    for n in range(1,order):
        vec += (1/n)*np.sin(x*n)
    return vec
#%%

def video(vid, fps=15, figsize=14, axis='auto'):
    assert len(vid.shape) == 3, "The dimension of the video's array should be equals 3"
    
    if axis == 'auto':
        inds = np.argsort(vid.shape)[::-1]
        inds = np.r_[inds[0], min(inds[1:]), max(inds[1:])]
        vid = vid.transpose(inds)
    else:
        inds = [0, 1, 2]
        inds.pop(axis)
        inds = np.append([axis], (inds)).tolist()

        vid = vid.transpose(inds)

    if fps > 60:
        maxframes = int(np.round(vid.shape[0]/fps * 60))
        inds = np.linspace(0, vid.shape[0]-1, maxframes)
        inds = inds.astype(int)
        vid = vid[inds]
        fps = 60
    
    plt.figure(figsize=(figsize, figsize*vid.shape[1]/vid.shape[2]))
    ax = plt.imshow(vid[0], vmin = vid.min(), vmax = vid.max())
    
    for frame in vid:
        ax.set_data(frame)
        plt.pause(1/fps)
#%%

# isig_filt = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_ICV/raw_signals_CBM_R1/DF_CBM_7.0p/isig_filt.mat')['isig_filt']
# isig_raw = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_ICV/raw_signals_CBM_R1/isig_HR.mat')['isig']


# isig_filt = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_LSPV/raw_signals_CBM_R1/DF_CBM_7.0p/isig_filt.mat')['isig_filt']
# isig_raw = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_LSPV/isig_HR.mat')['isig']


# isig_filt = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_RAA/raw_signals_CBM_R1/DF_CBM_7.0p/isig_filt.mat')['isig_filt']
# isig_raw = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_RAA/isig_HR.mat')['isig']


# isig_filt = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_RIPV/raw_signals_CBM_R1/DF_CBM_7.0p/isig_filt.mat')['isig_filt']
# isig_raw = sio.loadmat('/media/italo/HDD/AtrialArrythmia/Resultados de análises/HR/TA_RIPV/isig_HR.mat')['isig']

# #%%

# library_folder = os.path.join('/home','italo','scripts','vgmarques-bspm-atrial-arrhyhtmia-analysis')

# os.chdir(os.path.join(library_folder,'freq_analysis'))
# import frequencyAnalysis_lib as fqa

# os.chdir(os.path.join(library_folder,'phase_analysis'))
# import phaseAnalysis_lib as pha

# METHOD = 4
# fmin = 0.5
# fmax = 30
# df_range = [2,15]
# lims = [3,30]
# filter_option= 'SPATIAL'
# spat_filt = 0.07

# #Load Signals
# fs = 500 # Hz
 

# DF_wtmm, mode_wtmm, HDF_wtmm, RI_wtmm, OI_wtmm, isig_filt = fqa.completeDF_BSPM(isig_raw ,fs,fmin,fmax,
#                                                                           det_method=METHOD,
#                                                                           filter_option=filter_option, 
#                                                                           spat_filt=spat_filt,
#                                                                           band_oi=1, wav_lims=lims)


# fs_low = 128 #Hz

# # Remove begin and end of signal and downsample
# ms_to_remove = 200
# cut = int(ms_to_remove*fs_low*1e-3)
# isig_dec = sig.resample(isig_filt, int(isig_filt.shape[-1]/fs*fs_low), axis=2) #Downsampled

# # Phase with Hilbert transform
# Phase = pha.narrowBandHilbert(isig_dec, HDF_wtmm,fs_low,cut = ms_to_remove) 

#%%

fs = 500
nframes = 2000
sigfreq = 7
h, w = 128, 128

alpha = 0.01

# center = np.array([np.random.rand(1)*100, np.random.rand(1)*100])

for i, center in enumerate([np.array([32, 64]),
                            np.array([32, 80]),
                            np.array([32, 48]),
                            np.array([40, 32]),
                            np.array([64, 64])]):
    
    phaseprog = np.linspace(0,(nframes-1)/fs*sigfreq*2*np.pi, nframes)
    
    
    ys = np.arange(h)
    xs = np.arange(w)
    xx, yy = np.meshgrid(ys,xs)
    
    phasemap = np.sqrt((xx - center[0])**2 + (yy - center[1])**2) * 2*np.pi * alpha
    
    phasemap = phasemap.reshape(h,w,1) - phaseprog.reshape(1,1,-1)
    
    while np.any(phasemap>np.pi) or np.any(phasemap<=-np.pi):
        phasemap[phasemap>np.pi] -= 2*np.pi
        phasemap[phasemap<=-np.pi] += 2*np.pi
    
    signals = sawtooth(phasemap, 4)
    # np.save('/media/italo/7624C59E24C5622B/pesquisas/CinC2022/phantoms/ectopic' + str(i) + '.npy', signals)
    break