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

# change directory
path = '/Users/nigel.hussain/Documents' 
os.chdir(path)

# Access excel sheet. 
df = pd.read_csv('Therapy_Box.csv') # the sa1 graphs

# Extract the features and place them into Input dataframe
Input = df.iloc[:, 2:5]

# Convert to Array
x = Input.values
print(x)

# Train a Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)

# Estimate parameters
gmm.fit(Input)

# Predict results
gmm.predict(X) #  This was a different dataset (sa2).

# Visualise results (with 3d graph)
fig = pyplot.figure()
ax = Axes3D(fig) # convert to 3d axes (as we have three datapoints)

# create the graph
ax.scatter(x[:,0], x[:,1], x[:,2], c = labels, s=40, cmap='viridis')

# label axes
ax.set_xlabel('mean_mfcc')
ax.set_ylabel('mean_spectral_centroid')
ax.set_zlabel('mean_zero_crossing_rate')

pyplot.savefig('test.pdf') # save figure
pyplotshow() # visualise results 
