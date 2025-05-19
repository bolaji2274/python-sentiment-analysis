import librosa
import librosa.display
import matplotlib.pyplot as plt

file_path = 'Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav'
audio, sample_rate = librosa.load(file_path)

# Display waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sample_rate)
plt.title('Audio Waveform')
# plt.show()

import numpy as np

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    # mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    return np.concatenate((mfccs, chroma, mel))

features = extract_audio_features("Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")
print("Feature vector length:", len(features))

# import os
# import numpy as np
# import librosa

# # Function to extract features
# def extract_audio_features(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#     chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
#     mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
#     return np.concatenate((mfccs, chroma, mel))

# # Path to your base directory
# base_dir = "Audio_Speech_Actors_01-24"

# # Prepare lists for features and labels
# features_list = []
# labels_list = []

# # Loop through each actor folder
# for actor_folder in os.listdir(base_dir):
#     actor_path = os.path.join(base_dir, actor_folder)
    
#     if os.path.isdir(actor_path):
#         for filename in os.listdir(actor_path):
#             if filename.endswith(".wav"):
#                 file_path = os.path.join(actor_path, filename)

#                 # Extract features
#                 try:
#                     features = extract_audio_features(file_path)
#                     features_list.append(features)
                    
#                     # Optional: parse emotion label from filename (example: '03-01-03-01-01-01-01.wav')
#                     # The 3rd number (index 2) in the filename represents the emotion code
#                     emotion_code = int(filename.split("-")[2])
#                     labels_list.append(emotion_code)

#                 except Exception as e:
#                     print(f"Error processing {file_path}: {e}")

# print("Total files processed:", len(features_list))

import os
import numpy as np
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_audio_features(file_path):
    # load + features exactly as before
    y, sr = librosa.load(file_path, sr=None)
    mfccs  = np.mean(librosa.feature.mfcc(     y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T,        axis=0)
    mel    = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,     axis=0)
    return np.concatenate((mfccs, chroma, mel))

def worker(args):
    file_path, emotion_map = args
    feats = extract_audio_features(file_path)
    # parse label out of filename
    code = int(os.path.basename(file_path).split('-')[2])
    label = emotion_map.get(code, None)
    return feats, label

def parallel_feature_extraction(base_dir, emotion_map, max_workers=None):
    # gather all .wav paths
    files = []
    for actor in os.listdir(base_dir):
        actor_dir = os.path.join(base_dir, actor)
        if os.path.isdir(actor_dir):
            for fn in os.listdir(actor_dir):
                if fn.endswith('.wav'):
                    files.append(os.path.join(actor_dir, fn))

    features_list, labels_list = [], []
    # spin up a pool of processes
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # submit all jobs
        futures = {pool.submit(worker, (fp, emotion_map)): fp for fp in files}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                feats, label = future.result()
                features_list.append(feats)
                labels_list.append(label)
            except Exception as e:
                print(f"Failed {fp}: {e}")

    return np.array(features_list), np.array(labels_list)

if __name__ == "__main__":
    base_dir = "Audio_Speech_Actors_01-24"
    emotion_map = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
    }
    X, y = parallel_feature_extraction(base_dir, emotion_map, max_workers=8)
    print("Processed:", X.shape[0], "files;", "Feature dim:", X.shape[1])


import speech_recognition as sr

def audio_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "[Unrecognizable]"
    except sr.RequestError:
        return "[API Error]"

text = audio_to_text("Audio_Speech_Actors_01-24/Actor_01/03-01-03-01-01-01-01.wav")
print("Transcribed Text:", text)

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove digits
    return text.strip()

cleaned = clean_text(text)
print("Cleaned Text:", cleaned)

import pandas as pd

data = {
    'filename': [file_path],
    'transcript': [cleaned],
    'label': ['happy'],  # This will come from folder or dataset label
    'features': [features]
}

df = pd.DataFrame(data)
print(df.head())


