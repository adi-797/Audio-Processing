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
* Function Name: onset_detection
* Input: sound - time series array of the audio
* Output: onsets - list of the detected onsets
* Logic: This function breaks the array into frames and calculates their average amplitude and convert that into dBs, further
        a series of comparisons is done to obtain the onsets and to eliminate the hoax values.
* Example Call: onset_detection(sound)
'''
def onset_detection(sound):
    
    index = 0
    onsets_index = 0
    previous_amp = -1
    onsets = []
    
    while index < (len(sound)-int(441/2)):

        new_amp = 20*math.log10(sum(abs(sound[index : index + int(441)])))
        
        if previous_amp < 8.8 and new_amp > 8.8:
            
            check_index = index
            
            check_index += int(441/2)
            first_check = 20*math.log10(sum(abs(sound[check_index : check_index+int(441)])))

            check_index += int(441/2)
            second_check = 20*math.log10(sum(abs(sound[check_index : check_index + int(441)])))

            check_index += int(441/2)
            third_check = 20*math.log10(sum(abs(sound[check_index : check_index + int(441)])))

            if first_check > 8.8 and second_check > 8.9 and third_check > 9:
                onsets.append(round(float(onsets_index*(0.005)),2))

        previous_amp = new_amp
        index+=int(441/2)
        onsets_index+=1
        
    index = 0
    len_onsets = len(onsets)-1
    
    while index < len_onsets:
        
        if abs(onsets[index]-onsets[index+1]) < 0.07:
            
            onsets.pop(index)
            len_onsets = len(onsets)-1
            
        index+=1

    return onsets