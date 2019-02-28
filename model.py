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


def Instrument_detect(fs, sound):
    df = pd.read_csv('features.csv', index_col = 0)
    collist = df.columns.tolist()
    collist.remove('label')
    X = np.array(df[collist])
    y = np.array(df[['label']])


##  This section has been commented out as it is no longer required.
##  Helps in deciding the best features among the availabe set.

##    model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
##    rfe = RFE(model,50)
##    fit = rfe.fit(X, y.ravel())
##    print("Num Features: %d") % fit.n_features_
##    print("Selected Features: %s") % fit.support_
##    print collist
##    print("Feature Ranking: %s") % fit.ranking_


    # Removes the unnecessary features from the set.
    # collist.remove('bandwidth-var')
    # collist.remove('cqt-var')
    # collist.remove('flatness-mean')
    # collist.remove('flatness-var')
    # collist.remove('flux-var')
    # collist.remove('zcr-mean')
    # collist.remove('zcr-var')
    # collist.remove('tonal-centroid-mean')
    # collist.remove('tonal-centroid-var')
    # collist.remove('spectral-centroid-var')
    # collist.remove('rolloff-var')
    # collist.remove('rms-mean')
    # collist.remove('rms-var')
    # collist.remove('mfcc-var-3')
    # collist.remove('flux-mean')
    # collist.remove('cqt-mean')

    # Creates the vector for prediction.
    X = np.array(df[collist])


##  This region has been commented out as it no longer required for predicting the instrument.
##  Calculates the accuracy for different model using different split sizes and suggests the best possible combination.
##
##    ts = 0.01
##    mlp_acc = []
##    log_acc =[]
##    iterations = 1
##    
##    for i in range(iterations):
##        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=1)
##        ##print X_train.shape, X_test.shape,y_train.shape, y_test.shape
##        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
##        
##        clf = MLPClassifier(solver='sgd',max_iter =1000,verbose =False, alpha=1e-5,hidden_layer_sizes=(73, 1), random_state=1)
##        clf.fit(X_train, y_train.ravel())
##        pred_mlp = clf.predict(X_test)
##        mlp_acc.append(accuracy_score(y_test,pred_mlp))
##
##        trained = 'trained_instrument2.pkl'
##        decision_tree_model_pkl = open(trained, 'wb')
##        pickle.dump(clf, decision_tree_model_pkl)
##        decision_tree_model_pkl.close()
##
##        clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
##        clf.fit(X_train, y_train.ravel())
##
##        pred_log = clf.predict(X_test)
##        log_acc.append(accuracy_score(y_test,pred_log))
##
##        ts+=0.01
##
##    print "accuracy = ", log_acc, max(log_acc)#, max(mlp_acc), "mlp", mlp_acc,
##    best = 0.01 + 0.01*np.argmax(np.array(log_acc))

##    base = [b for b in range(iterations)]
##    print "accuracy = ", pred_log, "mlp", mlp_acc, max(pred_log)#, max(mlp_acc)
##    print 0.01 + 0.01*np.argmax(np.array(log_acc))
##    print "average = ", sum(log_acc)/len(log_acc)
##    plt.plot(base,mlp_acc,'y')
##    plt.plot(base,log_acc,'b')
##    plt.show()


    # Splits the data, 0.1 has returned an accuracy of 97%.
##    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1, random_state=1)
##    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
##
##    X_train, y_train = shuffle(X_train, y_train)
##
##
##    clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
##    clf.fit(X_train, y_train.ravel())
##
##
##    # Stores the trained model as a pickle binary file.
##    trained = 'trained_instrument.pkl'
##    decision_tree_model_pkl = open(trained, 'wb')
##    pickle.dump(clf, decision_tree_model_pkl)
##    decision_tree_model_pkl.close()


    # Loads the pre trained model
    decision_tree_model_pkl = open('trained_instrument.pkl', 'rb')
    clf = pickle.load(decision_tree_model_pkl)


    # Extracts all the features and removes the un necessary ones
    feature_set = extract_features(sound, fs)
    
    # del feature_set['bandwidth-var']
    # del feature_set['cqt-var']
    # del feature_set['flatness-mean']
    # del feature_set['flatness-var']
    # del feature_set['flux-var']
    # del feature_set['zcr-mean']
    # del feature_set['zcr-var']
    # del feature_set['flux-mean']
    # del feature_set['rms-var']
    # del feature_set['rms-mean']
    # del feature_set['cqt-mean']
    # del feature_set['mfcc-var-3']
    # del feature_set['tonal-centroid-mean']
    # del feature_set['tonal-centroid-var']
    # del feature_set['spectral-centroid-var']
    # del feature_set['rolloff-var'] 

    testing_vars = np.array(feature_set.values())

    # Predicts the instrument
    instrument_id = clf.predict(testing_vars.reshape(1, -1))

    return instrument_id