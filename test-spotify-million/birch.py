import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import Birch, AgglomerativeClustering, SpectralClustering, OPTICS, DBSCAN




import birchtools

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('spotify_data.csv')

subset_tracks = tracks.sample(n=50000, random_state=1)

# Save the subset to a new CSV file
subset_tracks.to_csv('spotify_subset.csv', index=False)

tracks = subset_tracks


tracks['artist_name'] = tracks['artist_name'].str.strip().str.lower()
tracks['track_name'] = tracks['track_name'].str.strip().str.lower()


# print(tracks.columns.tolist())

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
tracks['Birch_label'] = -1


# Turns artist_name and track_name into integers

tracks = tracks.drop(columns=['track_id', 'Unnamed: 0'])

print('=========================================')
scaler = StandardScaler()



# Sample the dataset for a quicker run
sample_size = 50000  # Adjust sample size as needed
if len(tracks) > sample_size:
    tracks_sample = tracks.sample(sample_size, random_state=42)
else:
    print("Sample size is larger than the dataset size. Running on the entire dataset.")
    tracks_sample = tracks

print("Sampled Tracks Shape:", tracks_sample.shape)

# Initialize the Birch model
model = Birch(n_clusters=10)
model.fit(tracks_sample)

# # Incremental fitting of the model
# for i in range(n_batches):
#     start_idx = i * batch_size
#     end_idx = min((i + 1) * batch_size, len(tracks))
#     mini_batch = tracks[start_idx:end_idx]
#     model.partial_fit(mini_batch)

# # Predict cluster labels for the entire dataset
tracks['Birch_label'] = model.predict(tracks)

print("Birch Labels:", tracks['Birch_label'].unique()
)


for _ in range(10):
    input_song_name = input("Enter a song name: \n")
    input_artist_name = input("Who is it by?\n")

    input_song = birchtools.find_song(
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

        birchtools.find_similar_songs(input_song_dict, tracks, model, scaler, le_artist, le_track)

        similar_songs = birchtools.find_similar_songs(input_song_dict, tracks, model, scaler, le_artist, le_track)
        similar_songs = similar_songs[1:10]
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

