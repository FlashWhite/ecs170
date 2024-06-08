import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

import knntools

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('../spotify_data.csv')

tracks['artist_name'] = tracks['artist_name'].str.strip().str.lower()
tracks['track_name'] = tracks['track_name'].str.strip().str.lower()


print(tracks.columns.tolist())

tracks.drop_duplicates()
tracks.dropna()

# One-Hot Encoding Approach (TOO MEMORY COSTLY)
# tracks = pd.get_dummies(tracks, columns=['artist_name', 'track_name'])

tracks = pd.get_dummies(tracks, columns=['genre'])


# Label Encoding
le_artist = LabelEncoder()
le_track = LabelEncoder()

tracks['artist_name'] = le_artist.fit_transform(tracks['artist_name'])
tracks['track_name'] = le_track.fit_transform(tracks['track_name'])

tracks = tracks.drop(columns=['track_id', 'Unnamed: 0'])

print('=========================================')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(tracks)

scaled_tracks = pd.DataFrame(scaled_features, columns=tracks.columns)

# for i in tracks.columns.tolist():
#     if "genre" in i:
#         print(i)


k = 11

knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
knn.fit(scaled_tracks)

for _ in range(10):
    input_song_name = input("Enter a song name: \n")
    input_artist_name = input("Who is it by?\n")

    input_song = knntools.find_song(
        input_artist_name.strip().lower(),
        input_song_name.strip().lower(),
        tracks,
        le_artist,
        le_track
    )

    if isinstance(input_song, str):
        print("Song not found")

    else:
        # Retrieve the actual songs
        input_song_dict = input_song.iloc[0].to_dict()
        similar_songs = knntools.find_similar_songs(input_song_dict, tracks, knn, scaler, le_artist, le_track)
        print("Here's a few songs I recommend you listen to!")
        print("=============================================")
        i = True
        for index, row in similar_songs.iterrows():
            # Lazy way of running first iteration on separate logic, doing this because knn returns original song too.
            if i:
                i = False
            else:
                print(f"'{row['track_name']}', by {row['artist_name']}\n")

    print()
    print()