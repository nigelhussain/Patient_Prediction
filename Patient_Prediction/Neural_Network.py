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

# change directory to where excel file is located
path = '/Users/nigel.hussain/Documents' 
os.chdir(path)

# Convert csv to pandas dataframe 
df = pd.read_csv('Therapy_Box.csv')

# Extract the features and place them into Input dataframe
Input = df.iloc[:, 2:5]
Output = df.Gender

# Build the model (neural network):
Neural_Network = MLPClassifier(hidden_layer_sizes=(3, ), 
                    activation='identity',
                    solver='sgd',
                    alpha=0, 
                    learning_rate='constant',
                    learning_rate_init=0.3, 
                    max_iter=100,
                    shuffle=False, 
                    random_state=None,
                    tol=0.0001, 
                    verbose=True, 
                    warm_start=False, 
                    momentum=0, 
                    nesterovs_momentum=False, 
                    early_stopping=False, 
                    validation_fraction=0)


# Train the Neural Network (Estimate parameters)
Neural_Network.train(Input, Output)

# Feed new dataset:
Test2 # example 
       mfcc      spectral_centroid     zero_crossing_rate
    -6.396145        1945.684432            0.099718

# Classify using the model:
predictions = Neural_Network.predictions(Test)

# This was classified as a male in my test.

# Performance metrics:
print(Neural_Network.predictions(Test))
