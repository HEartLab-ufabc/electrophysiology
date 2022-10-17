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

def angle_range(angs):
    while np.any(angs >= np.pi) or np.any(angs < -np.pi):
        angs[angs >= np.pi] -= 2*np.pi
        angs[angs < -np.pi] += 2*np.pi
    return angs

def sawtooth(x, order):
    vec = -np.sin(x)
    for n in range(1,order):
        vec -= (1/n)*np.sin(x*n)
    return vec
#%%

def video(vid, fps=15, figsize=14):
    assert len(vid.shape) == 3, "The dimension of the video's array should be equals 3"
    
    inds = np.argsort(vid.shape)[::-1]
    inds = np.r_[inds[0], min(inds[1:]), max(inds[1:])]
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

fs = 500 #Frequency of sampling
nframes = 2000 #Total number of frames
sigfreq = 8 #Rotor signal dominant frequency
h, w = 128, 128 #Size of the map
cicles = 0 #Number of cicles of the rotor epycenter around the map center


# Phase progression of the rotational activity, based on the number
# of framse (nframes) the frequency of sampling (fs) and the rotor 
# signal frequency (sigfreq).
phaseprog = np.linspace(0,(nframes-1)/fs*sigfreq*2*np.pi, nframes)

# The cw and ccw rotors are simetrically placed, and has its displacement
# speed increassing in the 3th order.
ang_position = np.linspace(0, 1, nframes)**4 * cicles*np.pi*2
# cwcenter = np.array([np.sin(ang_position)*(h/4) + (h/2), np.cos(ang_position)*(h/4) + (h/2)]).T
# ccwcenter = np.array([np.sin(ang_position + np.pi)*(h/4) + (h/2), np.cos(ang_position + np.pi)*(h/4) + (h/2)]).T

for i, pair in enumerate([np.array([[64, 96], [64, 32]]),
                          np.array([[64, 80], [64, 48]]),
                          np.array([[80, 96], [80, 32]]),
                          np.array([[80, 80], [80, 48]]),
                          np.array([[96, 96], [32, 32]])]):

    
    cwcenter = np.repeat(pair[(0,),], nframes, axis=0)
    ccwcenter = np.repeat(pair[(1,),], nframes, axis=0)
    # Creating xy grid
    ys = np.arange(h)
    xs = np.arange(w)
    xx, yy = np.meshgrid(ys,xs)
    
    # x and y relative positions from cw and ccw centers, in each frame.
    # dimensions: (xth position on grid, yth position on grid, time, relative x/y)
    # dimension
    cw_relat = np.concatenate((xx.reshape(h, w, 1, 1) - cwcenter[:,0].reshape(1, 1, -1, 1),
                               yy.reshape(h, w, 1, 1) - cwcenter[:,1].reshape(1, 1, -1, 1)),
                              axis = -1)
    ccw_relat = np.concatenate((xx.reshape(h, w, 1, 1) - ccwcenter[:,0].reshape(1, 1, -1, 1),
                                yy.reshape(h, w, 1, 1) - ccwcenter[:,1].reshape(1, 1, -1, 1)),
                               axis = -1)
    
    # Relative angle (which is the signal phase)
    cw_phase = angle_range(np.arctan2(cw_relat[...,0], cw_relat[...,1]) 
                           - ang_position.reshape(1,1,-1))
    ccw_phase = angle_range(-np.arctan2(ccw_relat[...,0], ccw_relat[...,1])
                            - np.pi + ang_position.reshape(1,1,-1))  # needed this correction to kepp simetry
    # Putting both in same angle refferencial
    ccw_phase[(cw_phase > np.pi/2)*(ccw_phase < -np.pi/2)] += 2*np.pi
    cw_phase[(cw_phase < -np.pi/2)*(ccw_phase > np.pi/2)] += 2*np.pi
    
    
    # Projection of ccw_center -> pixel on ccw_center -> cw_center. This is,
    # scalar product of ccw_relat and cwcenter - ccwcenter
    ccw_cw = (cwcenter - ccwcenter).reshape(1, 1, -1, 2)
    ccw_cw_proj = np.sum(ccw_relat * ccw_cw, axis=-1)/np.sum(ccw_cw**2, axis=-1)
    ccw_cw_proj[ccw_cw_proj>1] = 1
    ccw_cw_proj[ccw_cw_proj<0] = 0
    
    
    phasemap = ccw_cw_proj*cw_phase + (1-ccw_cw_proj)*ccw_phase
    
    phasemap = angle_range(phasemap.reshape(h,w,-1) - phaseprog.reshape(1,1,-1))
    
    
    signals = sawtooth(phasemap, 4)
    
    np.save('/media/italo/7624C59E24C5622B/pesquisas/CinC2022/phantoms/configs_stable' + str(i) + '.npy', {'fs':fs,
                                                                         'nframes':nframes,
                                                                         'shape':(h,w),
                                                                         'cicles':cicles})
    np.save('/media/italo/7624C59E24C5622B/pesquisas/CinC2022/phantoms/rotor_stable' + str(i) + '.npy', signals)
    np.save('/media/italo/7624C59E24C5622B/pesquisas/CinC2022/phantoms/SPs_stable' + str(i) + '.npy', {'cwcenter':cwcenter,
                                                                     'ccwcenter':ccwcenter})
