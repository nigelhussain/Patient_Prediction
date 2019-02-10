# Import the correct python folders (for the entire workload):
from __future__ import division
import librosa as librosa
import pandas as pd
import numpy as np
import librosa as librosa
import scipy.io.wavfile as wav
import pandas as pd
import scipy as sp
import os
import glob
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from numpy import fft as fft
from scipy.signal import butter, lfilter, freqz

file_paths = glob.glob('/Users/nigel.hussain/Desktop/Therapy/[1-9][1-9]/audio/*.wav')

aud_Data = []

for files in file_paths:
    rate, audData = wav.read(files, 'r')
    aud_Data.append(audData)
    
    
    
    
    
