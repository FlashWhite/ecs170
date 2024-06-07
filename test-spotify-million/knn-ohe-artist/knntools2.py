import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def find_song(artist_name, track_name, df, le_artist, le_track):
    try:
        artist_name_encoded = le_artist.transform([artist_name.lower().strip()])[0]
        track_name_encoded = le_track.transform([track_name.lower().strip()])[0]
    except ValueError:
        return "Song not found in the dataset."

    song = df[(df['artist_name'] == artist_name_encoded) & (df['track_name'] == track_name_encoded)]
    
    if not song.empty:
        song.loc[:, 'artist_name'] = le_artist.inverse_transform(song['artist_name']).astype(str)
        song.loc[:, 'track_name'] = le_track.inverse_transform(song['track_name']).astype(str)
        return song
    else:
        return "Song not found in the dataset."

def find_similar_songs(input_song, df, knn, scaler, le_artist, le_track):
    input_song_encoded = input_song.copy()
    input_song_encoded['artist_name'] = le_artist.transform([input_song_encoded['artist_name'].lower().strip()])[0]
    input_song_encoded['track_name'] = le_track.transform([input_song_encoded['track_name'].lower().strip()])[0]

    input_song_df = pd.DataFrame([input_song_encoded])
    
    # Drop 'cluster' column if present
    if 'cluster' in df.columns:
        df = df.drop(columns=['cluster'])
    
    # Reorder the columns to match the order used during fitting
    input_song_df = input_song_df[df.columns[:-1]]  # Exclude the last column ('cluster')
    
    # Add missing columns from the fit time
    for column in df.columns:
        if column not in input_song_df.columns:
            input_song_df[column] = 0

    input_song_scaled = scaler.transform(input_song_df)

    distances, indices = knn.kneighbors(input_song_scaled)
    nearest_neighbors = df.iloc[indices[0]].copy()

    nearest_neighbors['artist_name'] = le_artist.inverse_transform(nearest_neighbors['artist_name'])
    nearest_neighbors['track_name'] = le_track.inverse_transform(nearest_neighbors['track_name'])

    return nearest_neighbors
