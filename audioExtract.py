import numpy as np

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
    return np.concatenate((mfccs, chroma, mel))

features = extract_audio_features("audio_sample.wav")
print("Feature vector length:", len(features))
