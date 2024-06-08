import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors




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




def find_similar_songs(input_song, df, birch_model, scaler, le_artist, le_track):
    input_song_encoded = input_song.copy()
    # print(input_song_encoded)
    input_song_encoded['artist_name'] = le_artist.transform([input_song_encoded['artist_name'].lower().strip()])[0]
    input_song_encoded['track_name'] = le_track.transform([input_song_encoded['track_name'].lower().strip()])[0]

    input_song_df = pd.DataFrame([input_song_encoded])
    # input_song_scaled = scaler.transform(input_song_df)

    cluster_label = birch_model.predict(input_song_df)[0]
    nearest_neighbors = df[df['Birch_label'] == cluster_label].copy()

    # # Reverse scale the nearest_neighbors
    # nearest_neighbors_scaled = scaler.inverse_transform(nearest_neighbors)
    # # Decode the label-encoded values back to original
    nearest_neighbors['artist_name'] = le_artist.inverse_transform(nearest_neighbors['artist_name'])
    nearest_neighbors['track_name'] = le_track.inverse_transform(nearest_neighbors['track_name'])
    # print(nearest_neighbors)

    return nearest_neighbors
    