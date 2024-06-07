from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Feature columns
features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_ms', 'time_signature']

def recommend_songs_kmeans(inputTrackName, inputArtistName, tracks, n_recommendations=10):
    # standardize feature cols
    scaler = StandardScaler()
    scaledTracks = scaler.fit_transform(tracks[features])

    inputTrackName = inputTrackName.strip().lower()
    inputArtistName = inputArtistName.strip().lower()

    kmeans = KMeans(n_clusters=10, random_state=42)
    tracks['cluster'] = kmeans.fit_predict(scaledTracks)

    print(tracks.head())

    inputTrack = tracks[(tracks['track_name'].str.lower() == inputTrackName) & (tracks['artist_name'].str.lower() == inputArtistName)]

    if inputTrack.empty:
        print("Track not found in dataset")
        return []
    print("Input track found:")
    print(inputTrack)

    inputTrackFeatures = inputTrack[features]
    scaledInputTrackFeatures = scaler.transform(inputTrackFeatures)
    
    # Predict cluster for the input track
    inputCluster = kmeans.predict(scaledInputTrackFeatures)[0]
    print(f"Predicted cluster: {inputCluster}")
    
    # Get tracks from the same cluster
    similarTracks = tracks[tracks['cluster'] == inputCluster]
    # Exclude input track from recommendations
    similarTracks = similarTracks[(similarTracks['track_name'].str.lower() != inputTrackName) | (similarTracks['artist_name'].str.lower() != inputArtistName)]
    
    # Calculate the distances to the input track
    similarTracksFeatures = scaler.transform(similarTracks[features])
    distances = euclidean_distances(scaledInputTrackFeatures, similarTracksFeatures).flatten()
    # Add distances to the dataframe
    similarTracks['distance'] = distances
    
    # Sort by distance and select top 10
    recommendations = similarTracks.sort_values(by=['distance', 'popularity'], ascending=[True, False]).drop_duplicates(subset=['artist_name', 'track_name']).head(n_recommendations)
    if len(recommendations) < n_recommendations:
        # Fill remaining recommendations with similar tracks if duplicates
        remaining = similarTracks[~similarTracks.index.isin(recommendations.index)].head(n_recommendations - len(recommendations))
        recommendations = pd.concat([recommendations, remaining])


    print("Recommendations:")
    print(recommendations)