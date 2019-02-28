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
* Function Name: zcr
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(zcr_val) - mean of the zero crossings, np.var(zcr_val) - variance of the zero crossings
* Logic: This function is used to extract features of zero crossing rate by using Librosa library.
* Example Call: zcr(sound, 44100)
'''
def zcr(sound,fs):
    zcr_val = librosa.feature.zero_crossing_rate(sound)
    return np.mean(zcr_val), np.var(zcr_val)



'''
* Function Name: spectral_centroid
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(spectral_centroid) - mean of the spectral centroids, np.var(spectral_centroid) - variance of the spectral centroids
* Logic: This function is used to extract features of spectral centroid by using Librosa library.
* Example Call: spectral_centroid(sound, 44100)
'''
def spectral_centroid(sound,fs):
    spectral_centroid_val = librosa.feature.spectral_centroid(sound, fs)
    return np.mean(spectral_centroid_val), np.var(spectral_centroid_val)



'''
* Function Name: bandwidth
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(bandwidth) - mean of the bandwidths, np.var(bandwidth) - variance of the bandwidths
* Logic: This function is used to extract features of bandwidths by using Librosa library.
* Example Call: bandwidth(sound, 44100)
'''
def bandwidth(sound,fs):
    bandwidth_val = librosa.feature.spectral_bandwidth(sound, fs)
    return np.mean(bandwidth_val), np.var(bandwidth_val)



'''
* Function Name: mfcc
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: mfcc_std[1-13], mfcc_kurt[1-13], mfcc_skew[1-13], mfcc_var[1-13]
* Logic: This function is used to extract mfcc features (coefficients) by using Librosa library.
* Example Call: mfcc(sound, 44100)
'''
def mfcc(sound,fs):
    mfcc_val = librosa.feature.mfcc(sound, fs, n_mfcc= 13)
    mfcc_std = []
    mfcc_var = []
    mfcc_kurt = []
    mfcc_skew = []
    
    for i in range(mfcc_val.shape[0]):
        mfcc_std.append(np.std(mfcc_val[i]))
        mfcc_var.append(np.var(mfcc_val[i]))
        mfcc_kurt.append(scipy.stats.kurtosis(mfcc_val[i]))
        mfcc_skew.append(scipy.stats.skew(mfcc_val[i]))
        
    return mfcc_std[0],mfcc_std[1],mfcc_std[2],mfcc_std[3],mfcc_std[4],mfcc_std[5],mfcc_std[6],mfcc_std[7],mfcc_std[8],mfcc_std[9],mfcc_std[10],mfcc_std[11],mfcc_std[12],mfcc_var[0],mfcc_var[1],mfcc_var[2],mfcc_var[3],mfcc_var[4],mfcc_var[5],mfcc_var[6],mfcc_var[7],mfcc_var[8],mfcc_var[9],mfcc_var[10],mfcc_var[11],mfcc_var[12],mfcc_kurt[0],mfcc_kurt[1],mfcc_kurt[2],mfcc_kurt[3],mfcc_kurt[4],mfcc_kurt[5],mfcc_kurt[6],mfcc_kurt[7],mfcc_kurt[8],mfcc_kurt[9],mfcc_kurt[10],mfcc_kurt[11],mfcc_kurt[12],mfcc_skew[0],mfcc_skew[1],mfcc_skew[2],mfcc_skew[3],mfcc_skew[4],mfcc_skew[5],mfcc_skew[6],mfcc_skew[7],mfcc_skew[8],mfcc_skew[9],mfcc_skew[10],mfcc_skew[11],mfcc_skew[12]




'''
* Function Name: attack_time
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: attack_time - normalized time of attack (10% - 90%), attack_delta - time duration between end of attack and peak
* Logic: This function creates frames of the input array and stores the instant at which amplitude is 10% and then at 90%, the difference is the attack time,
         similarly, difference between the peak time and attack time is the attack delta
* Example Call: attack_time(sound, 44100)
'''
def attack_time(sound,fs):

    index = 0
    inc = int(fs/1000)
    envelopes = []
    attack_delta = 0.0
    peak_time = float(np.argmax(abs(sound)))/float(fs)
    peak_time = round(peak_time, 5)
    
    while index<len(sound):
        envelopes.append(sum(abs(sound[index : index+inc])))                
        index+=inc

    start_time = 0.0
    end_time = float(len(sound))/float(fs)
    index = 0
    maximum = max(envelopes)

    while index<len(envelopes):
        if envelopes[index] > (0.15*maximum) and envelopes[index] < (0.25*maximum):
            start_time = float(index*inc)/float(fs)
            
            while index<len(envelopes):
                if envelopes[index] > (0.85*maximum) and envelopes[index] < (0.95*maximum):
                    end_time = float(index*inc)/float(fs)
                    break
                index+=1
            break
        index+=1

    attack_time = ((end_time - start_time)/((float(len(sound))/float(fs))))
    attack_delta = peak_time - end_time
    attack_time = round(attack_time,5)
    attack_delta = round(attack_delta,5)
    
    return attack_time, attack_delta
                 


'''
* Function Name: flux
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(flux_val) - mean of the bandwidths, np.var(flux_val) - var of the bandwidths
* Logic: This function is used to extract features of flux by using Librosa library.
* Example Call: flux(sound, 44100)
'''
def flux(sound,fs):
    index = 0
    inc = int(fs/1000)
    eps = 0.00000001
    flux_val = []

    while index<(len(sound)-2*inc):
        X = abs(np.fft.fft(sound[index:index+inc]))
        X_prev = abs(np.fft.fft(sound[index+inc: index+2*inc]))
        sumX = numpy.sum(X + eps)
        sumPrevX = numpy.sum(X_prev + eps)
        flux_val.append(numpy.sum((X / sumX - X_prev/sumPrevX) ** 2))
        index+=inc
    
    return np.mean(flux_val), np.var(flux_val)



'''
* Function Name: rms
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(rms) - mean of the rms energies, np.var(rms) - var of the rms energies
* Logic: This function is used to extract features of rms energies by using Librosa library.
* Example Call: rms(sound, 44100)
'''
def rms(sound,fs):
    rms_val = librosa.feature.rmse(sound)
    return np.mean(rms_val), np.var(rms_val)



'''
* Function Name: rolloff
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(rolloff) - mean of the rolloffs, np.var(rolloff) - var of the rolloffs
* Logic: This function is used to extract features of rolloffs by using Librosa library.
* Example Call: rolloff(sound, 44100)
'''
def rolloff(sound,fs):
    rolloff_val = librosa.feature.spectral_rolloff(sound, fs)
    return np.mean(rolloff_val), np.var(rolloff_val)



'''
* Function Name: flatness
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(flatness) - mean of the flatness, np.var(flatness) - var of the flatness
* Logic: This function is used to extract features of flatness by using Librosa library.
* Example Call: flatness(sound, 44100)
'''
def flatness(sound,fs):
    flatness_val = librosa.feature.spectral_flatness(sound)
    return np.mean(flatness_val), np.var(flatness_val)



'''
* Function Name: cqt
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(cqt) - mean of the cqt, np.var(cqt) - var of the cqt
* Logic: This function is used to extract features of cqt by using Librosa library.
* Example Call: cqt(sound, 44100)
'''
def cqt(sound,fs):
    cqt_val = librosa.feature.chroma_stft(sound, fs,n_chroma=12, n_fft=4096)
    return np.mean(cqt_val), np.var(cqt_val)



'''
* Function Name: spectral_entropy
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: spectral_entropy_val - spectral entropy of the input signal
* Logic: This function calculates spectral entropy by using the defination of function defined in Pymir library
* Example Call: spectral_entropy(sound, 44100)
'''
def spectral_entropy(sound,fs):
    eps = 0.00000001
    n_short_blocks = 10
    L = len(sound)                      
    Eol = numpy.sum(sound ** 2)            

    sub_win_len = int(numpy.floor(L / n_short_blocks))   
    if L != sub_win_len * n_short_blocks:
        sound = sound[0:sub_win_len * n_short_blocks]

    sub_wins = sound.reshape(sub_win_len, n_short_blocks, order='F').copy()  
    s = numpy.sum(sub_wins ** 2, axis=0) / (Eol + eps)                     
    spectral_entropy_val = -numpy.sum(s*numpy.log2(s + eps))                        

    return spectral_entropy_val



'''
* Function Name: tonal_centroid
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: np.mean(tonal_centroid) - mean of the tonal_centroid, np.var(tonal_centroid) - var of the tonal_centroid
* Logic: This function is used to extract features of tonal_centroid by using Librosa library.
* Example Call: tonal_centroid(sound, 44100)
'''
def tonal_centroid(sound,fs):
    tonal_centroid_val = librosa.feature.tonnetz(sound, fs)
    return np.mean(tonal_centroid_val), np.var(tonal_centroid_val)



'''
* Function Name: extract_features
* Input: sound - time series array of the audio, fs - sampling frequency of the audio
* Output: feature - dictionary of all the features for training and prediction 
* Logic: This function calls other functions for extracting the features and storing them in a dictionary.
* Example Call: extract_features(sound, 44100)
'''
def extract_features(sound, fs):

    feature = {}

    feature['zcr-mean'],feature['zcr-var'] = zcr(sound,fs)

    feature['spectral-centroid-mean'], feature['spectral-centroid-var'] = spectral_centroid(sound,fs)
    
    feature['bandwidth-mean'], feature['bandwidth-var'] = bandwidth(sound,fs)

    feature['mfcc-std_dev-1'],feature['mfcc-std_dev-2'],feature['mfcc-std_dev-3'],feature['mfcc-std_dev-4'],feature['mfcc-std_dev-5'],feature['mfcc-std_dev-6'],feature['mfcc-std_dev-7'],feature['mfcc-std_dev-8'],feature['mfcc-std_dev-9'],feature['mfcc-std_dev-10'],feature['mfcc-std_dev-11'],feature['mfcc-std_dev-12'],feature['mfcc-std_dev-13'],feature['mfcc-var-1'],feature['mfcc-var-2'],feature['mfcc-var-3'],feature['mfcc-var-4'],feature['mfcc-var-5'],feature['mfcc-var-6'],feature['mfcc-var-7'],feature['mfcc-var-8'],feature['mfcc-var-9'],feature['mfcc-var-10'],feature['mfcc-var-11'],feature['mfcc-var-12'],feature['mfcc-var-13'],feature['mfcc-kurt-1'],feature['mfcc-kurt-2'],feature['mfcc-kurt-3'],feature['mfcc-kurt-4'],feature['mfcc-kurt-5'],feature['mfcc-kurt-6'],feature['mfcc-kurt-7'],feature['mfcc-kurt-8'],feature['mfcc-kurt-9'],feature['mfcc-kurt-10'],feature['mfcc-kurt-11'],feature['mfcc-kurt-12'],feature['mfcc-kurt-13'],feature['mfcc-skew-1'],feature['mfcc-skew-2'],feature['mfcc-skew-3'],feature['mfcc-skew-4'],feature['mfcc-skew-5'],feature['mfcc-skew-6'],feature['mfcc-skew-7'],feature['mfcc-skew-8'],feature['mfcc-skew-9'],feature['mfcc-skew-10'],feature['mfcc-skew-11'],feature['mfcc-skew-12'],feature['mfcc-skew-13'] = mfcc(sound,fs)

    feature['attack-time'], feature['attack-delta'] = attack_time(sound,fs)

    feature['flux-mean'],feature['flux-var'] = flux(sound,fs)

    feature['rms-mean'], feature['rms-var'] = rms(sound,fs)

    feature['rolloff-mean'], feature['rolloff-var'] = rolloff(sound,fs)

    feature['flatness-mean'], feature['flatness-var'] = flatness(sound,fs)

    feature['cqt-mean'], feature['cqt-var'] = cqt(sound,fs)

    feature['spectral-entropy'] = spectral_entropy(sound,fs)

    feature['tonal-centroid-mean'], feature['tonal-centroid-var'] = tonal_centroid(sound,fs)


##  This section has been commented out as it is not required for prediction once the csv file is generated.
##  For generating the csv file, pre defined label is attached to each file.
##
##    if len(file_name) == 13:    
##        if int(file_name[6:9]) in range(100,290):
##            feature['label'] = 1 #flute
##        elif int(file_name[6:9]) in range(290,431):
##            feature['label'] = 2 #violin
##        elif int(file_name[6:9]) in range(431,620):
##            feature['label'] = 3 #trumpet
##        elif int(file_name[6:9]) in range(620,1000):
##            feature['label'] = 4 #piano
##            
##    elif len(file_name)>13:
##        feature['label'] = 4 #piano
##        
##    else:
##        feature['label'] = 1 #flute

    return feature