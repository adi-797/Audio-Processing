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
* Function Name: f0_to_note_helper
* Input: f0 - fundamental frequency
* Output: note - corresponding note as string
* Logic: This function coverts the numeric value of f0 and returns the corresponding note using simple if else statements.
* Example Call: f0_to_note_helper(440)
'''
def f0_to_note_helper(f0):

    note = ""
    f0 = int(100*round(f0,2))
    
    if f0 in range(0,1699): #16.35
        note = 'C0'
    elif f0 in range(1699,1766): #17.32
        note = 'C#0'
    elif f0 in range(1766,1908): #18.35
        note = 'D0'
    elif f0 in range(1908,1983): #19.45
        note = 'D#0'
    elif f0 in range(1983,2121): #20.60
        note = 'E0'
    elif f0 in range(2121,2269): #21.83
        note = 'F0'
    elif f0 in range(2269,2358): #23.12
        note = 'F#0'
    elif f0 in range(2358,2547): #24.50
        note = 'G0'
    elif f0 in range(2547,2647): #25.96
        note = 'G#0'
    elif f0 in range(2647,2859): #27.50
        note = 'A0'
    elif f0 in range(2859,2971): #29.14
        note = 'A#0'
    elif f0 in range(2971,3178): #30.87
        note = 'B0'
    elif f0 in range(3178,3400): #32.70
        note = 'C1'
    elif f0 in range(3400,3533): #34.65
        note = 'C#1'
    elif f0 in range(3533,3816): #36.71
        note = 'D1'
    elif f0 in range(3816,3966): #38.86
        note = 'D#1'
    elif f0 in range(3966,4242): #41.20
        note = 'E1'
    elif f0 in range(4242,4538): #43.65
        note = 'F1'
    elif f0 in range(4538,4716): #46.25
        note = 'F#1'
    elif f0 in range(4716,5094): #49.00
        note = 'G1'
    elif f0 in range(5094,5294): #51.91
        note = 'G#1'
    elif f0 in range(5294,5718): #55.00
        note = 'A1'
    elif f0 in range(5718,5924): #58.27
        note = 'A#1'
    elif f0 in range(5924,6357): #61.74
        note = 'B1'
    elif f0 in range(6357,6800): #65.41
        note = 'C2'
    elif f0 in range(6800,7067): #69.30
        note = 'C#2'
    elif f0 in range(7067,7632): #73.42
        note = 'D2'
    elif f0 in range(7632,7932): #77.78
        note = 'D#2'
    elif f0 in range(7932,8486): #82.41
        note = 'E2'
    elif f0 in range(8486,9077): #87.31
        note = 'F2'
    elif f0 in range(9077,9433): #92.50
        note = 'F#2'
    elif f0 in range(9433,10188): #98.00
        note = 'G2'
    elif f0 in range(10188,10588): #103.83
        note = 'G#2'
    elif f0 in range(10588,11436): #110.00
        note = 'A2'
    elif f0 in range(11436,11885): #116.54
        note = 'A#2'
    elif f0 in range(11885,12714): #123.47
        note = 'B2'
    elif f0 in range(12714,13599): #130.81
        note = 'C3'
    elif f0 in range(13599,14133): #138.59
        note = 'C#3'
    elif f0 in range(14133,15265): #146.83
        note = 'D3'
    elif f0 in range(15265,15864): #155.56
        note = 'D#3'
    elif f0 in range(15864,16971): #164.81
        note = 'E3'
    elif f0 in range(16971,18153): #174.61
        note = 'F3'
    elif f0 in range(18153,18866): #185.00
        note = 'F#3'
    elif f0 in range(18866,20376): #196.00
        note = 'G3'
    elif f0 in range(20376,21176): #207.65
        note = 'G#3'
    elif f0 in range(21176,22872): #220.00
        note = 'A3'
    elif f0 in range(22872,23770): #233.08
        note = 'A#3'
    elif f0 in range(23770,25428): #246.94
        note = 'B3'
    elif f0 in range(25428,27199): #261.63
        note = 'C4'
    elif f0 in range(27199,28267): #277.18
        note = 'C#4'
    elif f0 in range(28267,30530): #293.66
        note = 'D4'
    elif f0 in range(30530,31729): #311.13
        note = 'D#4'
    elif f0 in range(31729,33943): #329.63
        note = 'E4'
    elif f0 in range(33943,36307): #349.23
        note = 'F4'
    elif f0 in range(36307,37732): #369.99
        note = 'F#4'
    elif f0 in range(37732,40753): #392.00
        note = 'G4'
    elif f0 in range(40753,42353): #415.30
        note = 'G#4'
    elif f0 in range(42353,45744): #440.00
        note = 'A4'
    elif f0 in range(45744,47540): #466.16
        note = 'A#4'
    elif f0 in range(47540,50856): #493.88
        note = 'B4'
    elif f0 in range(50856,54399): #523.25
        note = 'C5'
    elif f0 in range(54399,56535): #554.37
        note = 'C#5'
    elif f0 in range(56535,61061): #587.33
        note = 'D5'
    elif f0 in range(61061,63458): #622.25
        note = 'D#5'
    elif f0 in range(63458,67885): #659.25
        note = 'E5'
    elif f0 in range(67885,72614): #698.46
        note = 'F5'
    elif f0 in range(72614,75465): #739.99
        note = 'F#5'
    elif f0 in range(75465,81507): #783.99
        note = 'G5'
    elif f0 in range(81507,84707): #830.61
        note = 'G#5'
    elif f0 in range(84707,91488): #880.00
        note = 'A5'
    elif f0 in range(91488,95081): #932.33
        note = 'A#5'
    elif f0 in range(95081,101713): #987.77
        note = 'B5'
    elif f0 in range(101713,108798): #1046.50
        note = 'C6'
    elif f0 in range(108798,113070): #1108.73
        note = 'C#6'
    elif f0 in range(113070,122122): #1174.66
        note = 'D6'
    elif f0 in range(122122,126917): #1244.51
        note = 'D#6'
    elif f0 in range(126917,135771): #1318.51
        note = 'E6'
    elif f0 in range(135771,145229): #1396.96
        note = 'F6'
    elif f0 in range(145229,150931): #1479.98
        note = 'F#6'
    elif f0 in range(150931,163014): #1567.98
        note = 'G6'
    elif f0 in range(163014,169414): #1661.22
        note = 'G#6'
    elif f0 in range(163414,182977): #1760.00
        note = 'A6'
    elif f0 in range(182977,190161): #1864.66
        note = 'A#6'
    elif f0 in range(190161,203426): #1975.53
        note = 'B6'
    elif f0 in range(203426,217597): #2093.00
        note = 'C7'
    elif f0 in range(217597,226141): #2217.46
        note = 'C#7'
    elif f0 in range(226141,244245): #2349.32
        note = 'D7'
    elif f0 in range(244245,253835): #2489.02
        note = 'D#7'
    elif f0 in range(253835,271542): #2637.02
        note = 'E7'
    elif f0 in range(271542,290458): #2793.83
        note = 'F7'
    elif f0 in range(290458,301862): #2959.96
        note = 'F#7'
    elif f0 in range(301862,326028): #3135.96
        note = 'G7'
    elif f0 in range(326028,338829): #3322.44
        note = 'G#7'
    elif f0 in range(338829,367954): #3520.00
        note = 'A7'
    elif f0 in range(367954,382323): #3729.31
        note = 'A#7'
    elif f0 in range(382323,406854): #3951.07
        note = 'B7'
    elif f0 in range(406854,435195): #4186.01
        note = 'C8'
    elif f0 in range(435195,452282): #4434.92
        note = 'C#8'
    elif f0 in range(452282,488489): #4698.63
        note = 'D8'
    elif f0 in range(488489,507670): #4978.03
        note = 'D#8'
    elif f0 in range(507670,543084): #5274.04
        note = 'E8'
    elif f0 in range(543084,580915): #5587.65
        note = 'F8'
    elif f0 in range(580915,603725): #5919.91
        note = 'F#8'
    elif f0 in range(603725,652056): #6271.93
        note = 'G8'
    elif f0 in range(652056,677658): #6644.88
        note = 'G#8'
    elif f0 in range(677658,731908): #7040.00
        note = 'A8'
    elif f0 in range(731908,760645): #7458.62
        note = 'A#8'
    elif f0 in range(760645,1000000): #7902.13
        note = 'B8'
    else:
        note = 'ERROR: out of bounds'
        
    return note


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
