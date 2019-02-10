def feature_extraction(data):

    # this is a function that will return the 6 features that were decided upon and put them in a list

    # extract mean spectral centroid
    spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=data, sr = 22500).T,axis=0))

    # extract mean mfcc
    mfcc = np.mean(np.mean(librosa.feature.mfcc(y= data, sr = 22500, n_mfcc = 40).T,axis=0))

    # Extract mean zero crossing rate
    zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y= data).T,axis=0))

    # Extract mean spectral rolloff
    spectral_rolloff = np.mean(np.mean(librosa.feature.spectral_rolloff(y= data, sr = 22500).T,axis=0))

    # Return root mean square energy
    RMSE = np.mean(np.mean(librosa.feature.rmse(y= data).T,axis=0))

    # Return melspectrogram values
    melspectrogram = np.mean(np.mean(librosa.feature.melspectrogram(y= data, sr = 22500).T,axis=0))

    print(list((spec_cent, mfcc, zero_crossing_rate, spectral_rolloff, RMSE, melspectrogram)))

    
