import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np




def find_song(artist_name, track_name, df, le_artist, le_track):
    # Encode the input artist_name and track_name
    try:
        artist_name_encoded = le_artist.transform([artist_name.lower().strip()])[0]
        track_name_encoded = le_track.transform([track_name.lower().strip()])[0]
    except ValueError:
        return "Song not found in the dataset."

    # Search for the song in the dataset
    song = df[(df['artist_name'] == artist_name_encoded) & (df['track_name'] == track_name_encoded)]

    if not song.empty:
        # Decode the artist_name and track_name for readability
        song['artist_name'] = le_artist.inverse_transform(song['artist_name'])
        song['track_name'] = le_track.inverse_transform(song['track_name'])
        return song
    else:
        return "Song not found in the dataset."    

def get_random_song(df, le_artist, le_track):
    random_index = np.random.randint(0, len(df))
    random_song = df.iloc[random_index]

    # Decode the artist and track names
    artist_name = le_artist.inverse_transform([random_song['artist_name']])[0]
    track_name = le_track.inverse_transform([random_song['track_name']])[0]

    random_song = random_song.to_dict()
    random_song['artist_name'] = artist_name
    random_song['track_name'] = track_name

    return random_song


def find_similar_songs(input_song, df, knn, scaler, le_artist, le_track):
    input_song_encoded = input_song.copy()
    input_song_encoded['artist_name'] = le_artist.transform([input_song_encoded['artist_name'].lower().strip()])[0]
    input_song_encoded['track_name'] = le_track.transform([input_song_encoded['track_name'].lower().strip()])[0]

    input_song_df = pd.DataFrame([input_song_encoded])

    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])

    input_song_df = input_song_df[df.columns[:-1]]

    for column in df.columns:
        if column not in input_song_df.columns:
            input_song_df[column] = 0
            
    input_song_scaled = scaler.transform(input_song_df)

    distances, indices = knn.kneighbors(input_song_scaled)
    nearest_neighbors = df.iloc[indices[0]].copy()

    # Decode the label-encoded values back to original
    nearest_neighbors['artist_name'] = le_artist.inverse_transform(nearest_neighbors['artist_name'])
    nearest_neighbors['track_name'] = le_track.inverse_transform(nearest_neighbors['track_name'])

    return nearest_neighbors