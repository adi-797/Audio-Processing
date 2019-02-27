import numpy as np
import math
import wave
import os

import matplotlib.pyplot as plt
import serial, time
import struct, scipy, pandas as pd, sklearn, pyAudioAnalysis, librosa, aubio, pickle, numpy
from scipy.signal import blackmanharris, fftconvolve, find_peaks
from numpy.fft import rfft
from os import listdir
from os.path import isfile, join
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


'''
* Function Name: note_detect
* Input: sound - fs - sampling frequency of the audio, time series array of the audio
* Output: Detected_Note - the detected note for the audio
* Logic: This function first determines whether the waveform is pure sine or not by calculating the zero crossings and peaks,
         then for pure sine waves, f0 is calculated by fft method oterwise by autocorrelation.
* Example Call: note_detect(fs, sound)
'''
def note_detect(fs, sound):
    
    Detected_Note = ""
    file_length = len(sound)

    # Obtaining a sample for determining the peaks in the waveform
    # Calculating 'f0' by DFT method for pure sine waves, as 'f0' is generally stronger than harmonics in such waveforms
    # sample = sound[int(file_length/3) :]

    k = 0
    thresho = 0
    prev = 0
    
    while k < (file_length/2 - (fs/100)):
        thresho = sum([abs(sound[f]) for f in range(k,k+int(fs/100))])
        if thresho > prev:
            prev = thresho
            indices = k
        k+=int(fs/100)

    sample=sound[indices:]
    
    x= sample
    i=0
    zeros = []
    temp = (file_length/2)
    zero = sum(x)/len(x)

    for i in range(len(x)-1):
        if (x[i]>zero and x[i+1]<zero) or (x[i]<zero and x[i+1]>zero) or (x[i]>zero and x[i+1]==zero) or (x[i]<zero and x[i+1]==zero):
            zeros.append(i)
            if len(zeros)==30:
                temp = i
                break
            
    sample = sample[:temp+1]
    x = sample

    peaks, _ = find_peaks(sample, height=0)
    peaksneg, _ = find_peaks(-sample, height=0)   # For detecting negative/inverse peaks

    total_peaks = np.concatenate([peaks, peaksneg])
    total_peaks = sorted(total_peaks) # Indices of all the obtained peeks

    i = 0
    leng = len(total_peaks)
    while i<leng:
        if total_peaks[i]<zeros[0]:
            total_peaks.pop(i)
            leng-=1
        if total_peaks[i]>zeros[len(zeros)-1]:
            total_peaks.pop(i)
            leng-=1
        i+=1

    if len(total_peaks)-len(zeros) == -1:
        sine = True
    else:
        sine = False
    
    if sine is True:
        
        windowed = sound * blackmanharris(len(sound))
        f = rfft(windowed)
        i = np.argmax(abs(f))
        f = np.log(abs(f))
        x = i
        n = 100
        a, b, c = np.polyfit(np.arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
        px = -0.5 * b/a
        f0 = fs * px / len(windowed)

    # Calculating 'f0' by autocorrelation method as harmonics might be stronger than 'f0' in non-pure sine waveforms
    else:
        
        autocorr_array = fftconvolve(sound, sound[::-1], mode='full')                                
        autocorr_array = autocorr_array[len(autocorr_array)//2:]

        difference = np.diff(autocorr_array)
        first_low, = np.nonzero(np.ravel(difference>0))
        first_low = first_low[0]

        # finds the next peak
        peak = np.argmax(autocorr_array[first_low:]) + first_low
        peakx = 1/2. * (autocorr_array[peak-1] - autocorr_array[peak+1]) / (autocorr_array[peak-1] - 2 * autocorr_array[peak] + autocorr_array[peak+1]) + peak

        f0 = fs / peakx

    # Calling helper function to determine the note for given frequency
    Detected_Note = f0_to_note_helper(f0)

    return Detected_Note
