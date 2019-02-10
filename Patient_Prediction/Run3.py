
path = '/Users/nigel.hussain/Desktop/Therapy/8/audio'
os.chdir(path)
X, sample_rate = librosa.load('sa1.wav', res_type='kaiser_fast')
spec_cent = np.mean(np.mean(librosa.feature.spectral_centroid(y=X, sr = sample_rate).T,axis=0))
mfcc = np.mean(np.mean(librosa.feature.mfcc(y=X, sr = sample_rate, n_mfcc = 40).T,axis=0))
zero_crossing_rate = np.mean(np.mean(librosa.feature.zero_crossing_rate(y=X).T,axis=0))
d8 = [8, 'F', mfcc, spec_cent, zero_crossing_rate]
Input2 = [[mfcc, spec_cent, zero_crossing_rate]]
print(d8)
df = pd.DataFrame(data = Input2, columns=['mfcc', 'spectral_centroid', 'zero_crossing_rate'])
>>> Input2 = df.iloc[:, :]
Input2 = df.iloc[:, :]
Neural_Network.predict(Input2)
