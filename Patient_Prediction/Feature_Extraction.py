from __future__ import division
import wave
import librosa as librosa
import pandas as pd
import numpy as np
import librosa as librosa
import scipy.io.wavfile as wav
import pandas as pd
import scipy as sp
import librosa as librosa
import os
import sounddevice as sd
from numpy import fft as fft
from scipy.signal import butter, lfilter
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# Read the data into usable form

X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')

# return the rate
print(X)

# return the audData
print(sample_rate)

# return the mean mfcc, mean spectral centroid, and mean zero_crossing_rate for
# the data. This is a sample of what I did.

path = '/Users/nigel.hussain/Desktop/Therapy/1/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d1 = [1, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d1)

path = '/Users/nigel.hussain/Desktop/Therapy/2/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d2 = [2, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d2)

path = '/Users/nigel.hussain/Desktop/Therapy/3/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d3 = [3, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d3)

path = '/Users/nigel.hussain/Desktop/Therapy/4/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d4 = [4, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d4)

path = '/Users/nigel.hussain/Desktop/Therapy/5/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d5 = [5, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d5)

path = '/Users/nigel.hussain/Desktop/Therapy/6/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d6 = [6, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d6)

path = '/Users/nigel.hussain/Desktop/Therapy/7/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d7 = [7, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d7)


path = '/Users/nigel.hussain/Desktop/Therapy/8/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d8 = [8, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d8)

path = '/Users/nigel.hussain/Desktop/Therapy/9/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d9 = [9, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d9)

path = '/Users/nigel.hussain/Desktop/Therapy/10/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d10 = [10, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d10)

path = '/Users/nigel.hussain/Desktop/Therapy/11/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d11 = [11, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d11)

path = '/Users/nigel.hussain/Desktop/Therapy/12/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d12 = [12, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d12)

path = '/Users/nigel.hussain/Desktop/Therapy/13/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d13 = [13, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d13)

path = '/Users/nigel.hussain/Desktop/Therapy/14/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d14 = [14, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d14)

path = '/Users/nigel.hussain/Desktop/Therapy/15/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d15 = [15, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d15)

path = '/Users/nigel.hussain/Desktop/Therapy/16/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d16 = [16, 'F', mfcc, spec_cent, zero_crossing_rate]
print(d16)

path = '/Users/nigel.hussain/Desktop/Therapy/17/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d17 = [17, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d17)

path = '/Users/nigel.hussain/Desktop/Therapy/18/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d18 = [18, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d18)

path = '/Users/nigel.hussain/Desktop/Therapy/19/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d19 = [19, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d19)

path = '/Users/nigel.hussain/Desktop/Therapy/20/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d20 = [20, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d20)

path = '/Users/nigel.hussain/Desktop/Therapy/21/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d21 = [21, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d21)

path = '/Users/nigel.hussain/Desktop/Therapy/22/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d22 = [22, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d22)

path = '/Users/nigel.hussain/Desktop/Therapy/23/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d23 = [23, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d23)

path = '/Users/nigel.hussain/Desktop/Therapy/24/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d24 = [24, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d24)

path = '/Users/nigel.hussain/Desktop/Therapy/25/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d25 = [25, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d25)

path = '/Users/nigel.hussain/Desktop/Therapy/26/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d26 = [26, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d26)

path = '/Users/nigel.hussain/Desktop/Therapy/27/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d27 = [27, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d27)

path = '/Users/nigel.hussain/Desktop/Therapy/28/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d28 = [28, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d28)

path = '/Users/nigel.hussain/Desktop/Therapy/29/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d29 = [29, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d29)

path = '/Users/nigel.hussain/Desktop/Therapy/30/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d30 = [30, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d30)

path = '/Users/nigel.hussain/Desktop/Therapy/31/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d31 = [31, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d31)

path = '/Users/nigel.hussain/Desktop/Therapy/32/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d32 = [32, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d32)

path = '/Users/nigel.hussain/Desktop/Therapy/33/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d33 = [33, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d33)

path = '/Users/nigel.hussain/Desktop/Therapy/34/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d34 = [34, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d34)

path = '/Users/nigel.hussain/Desktop/Therapy/35/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d35 = [35, 'M', mfcc, spec_cent, zero_crossing_rate]
print(d35)

# These were then manually saved onto an excel sheet for efficiency and
# accuracy purposes.
