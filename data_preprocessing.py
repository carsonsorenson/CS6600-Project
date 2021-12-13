import pandas as pd
import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = './data/'
TRACKS_PATH = './data/tracks.csv'
AUDIO_FOLDER = './data/fma_small/'
SAMPLE_RATE = 8000
AUDIO_LENGTH = 40000

values = np.array(['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International', 'Electronic', 'Instrumental'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

def load_metadata(fpath):
    tracks = pd.read_csv(fpath, index_col=0, header=[0, 1])
    useful_columns = [('set', 'split'), ('set', 'subset'), ('track', 'genre_top')]
    df = tracks[useful_columns]
    df.columns = [col[1] for col in df.columns]
    df = df[df['subset'] == 'small']
    return df


def load_audio(fpath, tracks):
    d = {}
    cnt = 0
    for root, dirs, files in os.walk(fpath):
        for filename in files:
            track_id = filename.replace('.mp3', '')
            if filename.endswith('.mp3') and int(track_id) in list(tracks.index):
                track = tracks.loc[int(track_id)]
                split = track['split']
                genre = track['genre_top']
                if split not in d:
                    d[split] = []
                mp3 = root + '/' + filename
                audio, sr = librosa.load(mp3, sr=SAMPLE_RATE, mono=True)
                audio = audio[:AUDIO_LENGTH]
                d[split].append((audio, genre))
                cnt += 1
                if cnt % 100 == 0:
                    print("Loaded " + str(cnt) + " files")
    return d

def load_audio_spectogram(fpath, tracks):
    d = {}
    cnt = 0
    x = []
    for root, dirs, files in os.walk(fpath):
        for filename in files:
            track_id = filename.replace('.mp3', '')
            if filename.endswith('.mp3') and int(track_id) in list(tracks.index):
                track = tracks.loc[int(track_id)]
                split = track['split']
                genre = track['genre_top']
                if split not in d:
                    d[split] = []
                mp3 = root + '/' + filename
                y, sr = librosa.load(mp3)
                # get the first 20 seconds (if I do 30 I get slightly different sizes because of the different sized audio files)
                y = y[:(sr * 20)]
                spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, n_mels=64)
                spect = librosa.power_to_db(spect, ref=np.max)
                d[split].append((spect, genre))
                cnt += 1
                if cnt % 10 == 0:
                    print("Loaded " + str(cnt) + " files")
    return d


def encode_labels(labels):
    labels = np.array(labels)
    integer_encoded = label_encoder.transform(labels).reshape(-1, 1)
    return onehot_encoder.transform(integer_encoded)

def save_data(data, prefix=""):
    train_x = np.array([tup[0] for tup in data['training']])
    train_y = encode_labels([tup[1] for tup in data['training']])
    test_x = np.array([tup[0] for tup in data['test']])
    test_y = encode_labels([tup[1] for tup in data['test']])
    valid_x = np.array([tup[0] for tup in data['validation']])
    valid_y = encode_labels([tup[1] for tup in data['validation']])

    with open(os.path.join(DATA_PATH, prefix + "train_x.pck"), "wb") as o:
        pickle.dump(train_x, o)

    with open(os.path.join(DATA_PATH, prefix + "train_y.pck"), "wb") as o:
        pickle.dump(train_y, o)

    with open(os.path.join(DATA_PATH, prefix + "test_x.pck"), "wb") as o:
        pickle.dump(test_x, o)

    with open(os.path.join(DATA_PATH, prefix + "test_y.pck"), "wb") as o:
        pickle.dump(test_y, o)

    with open(os.path.join(DATA_PATH, prefix + "valid_x.pck"), "wb") as o:
        pickle.dump(valid_x, o)

    with open(os.path.join(DATA_PATH, prefix + "valid_y.pck"), "wb") as o:
        pickle.dump(valid_y, o)


def main():
    tracks = load_metadata(TRACKS_PATH)
    data = load_audio(AUDIO_FOLDER, tracks)
    save_data(data)
    #spec_data = load_audio_spectogram(AUDIO_FOLDER, tracks)
    #save_data(spec_data, prefix='spec_')


if __name__ == '__main__':
    main()
