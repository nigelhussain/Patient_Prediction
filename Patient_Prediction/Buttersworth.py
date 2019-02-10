# Import the correct python folders:
from __future__ import division
import librosa as librosa
import pandas as pd
import numpy as np
import librosa as librosa
import scipy.io.wavfile as wav
import pandas as pd
import scipy as sp
import os
import sounddevice as sd
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from numpy import fft as fft
from scipy.signal import butter, lfilter


# I renamed the folder from Therapy Box Test Data to Therapy to make the 'os' change a bit easier. 
# change the path
path = '/Users/nigel.hussain/Desktop/Therapy/1/audio' # The 1 is just an example. It can be anyone of the folder names
os.chdir(path)

# Step 1: Read in the audio file
rate, audData = wave.read('sa1.wav', 'r')

# Step 2: playback the audio file
sd.play(audData, rate)

# Step 3: perform fourier transformation to analyse noise frequency
fourier = fft.fftshift(audData) 
# The data appears to be uncentered by default, so the fftshift will center the data to increase readabiblity

# Step 4 plot the absolute function to determine magnitude of the signal. This will give us an idea of the contribution of 
# each frequency to the audio sample.
plt.plot(abs(fourier))

# Step 5: Implementing the Band-Pass filter

# calculate upper and lower bounds. This unfortunately can only be determined through trial and error 
# of each sample (300 and 3000 are placeholders, as they represent the lowest and highest 
# bounds of the detectable human voice signal)

lo,hi=300,3000 


b,a = butter(N=6, Wn=[2*lo/rate, 2*hi/rate], btype='band') # create the band-pass filter

x =lfilter(b,a,audData) # filter the signal using results from band-pass filter

# playback the filtered sound
sd.play(x, rate)

#Save the new audio file
sf.write('sa1.wav', x, rate)
