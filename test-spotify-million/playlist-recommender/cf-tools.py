import csv
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import os
from dotenv import load_dotenv
import time

def recommendSongs(user_playlist: pd.DataFrame, k=5, n_recs=10) -> pd.DataFrame:
    """
    Recommends `n_rec` songs given an input playlist, no very big playlists
    
    Parameters:
        * user_playlist: DataFrame of Tracks (from csv) without features 
        * k: Number of clusters
        * n_recs: Number of recommendations
    """
    # Fetch environment variables
    load_dotenv()
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')

    # Set up Spotify API credentials
    client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    # ------------------------------ GET USER PLAYLIST DATA ------------------------------
    features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
    user_tracks = [] # Output of list of track dicts
    track_info_100 = [] # Intermediate list for storing track info
    track_ids_100 = [] # Intermediate list for storing track ids
    for index, row in user_playlist.iterrows():
        id = row['Track URI'][14:]
        track_ids_100.append(id)
        track_info = {
            '':index,
            'artist_name': row['Artist Name(s)'],
            'track_name': row['Track Name'],
            'track_id': id,
            # ... we will add more attributes later
        }
        track_info_100.append(track_info)

        if len(track_ids_100) == 100 or index == len(user_playlist) - 1:
            response = sp.audio_features(track_ids_100)
            print(f"Successful response with {len(response)} entries")
            # Update track info for each song
            for response_index, audio_features in enumerate(response):
                if audio_features:  # Check if audio_features is not None
                    for feature in features:
                        track_info_100[response_index][feature] = audio_features[feature]
                    user_tracks.append(track_info_100[response_index])
            # Reset intermediate variables
            print(f"Total songs stored so far: {len(user_tracks) = }")
            track_ids_100 = []
            track_info_100 = []
            time.sleep(2)
    user_df = pd.DataFrame(user_tracks) # Convert to DataFrame
    # -------------------- USER CENTROID --------------------
    # Exclude track name, artist name, track id, index,...
    user_features = user_df[features] 
    centroid_vec = user_features.mean(axis=0)
    user_centroid = pd.DataFrame(centroid_vec).transpose()
    # -------------------- CLUSTER PLAYLISTS --------------------
    centroids_df = pd.read_csv('centroids_with_clusters.csv')
    # Drop the 'cluster' column from centroids_df to exclude it from the input
    if 'cluster' in centroids_df.columns:
        centroids_df = centroids_df.drop(columns=['cluster'])
    kmeans = KMeans(n_clusters=k)
    # Standardize Feature Columns
    scaler = StandardScaler()
    scaled_centroids_df = scaler.fit_transform(centroids_df[features])
    # KMeans Cluster
    kmeans.fit(scaled_centroids_df)
    # -------------------- PREDICT USER PLAYLIST CLUSTER --------------------
    user_cluster = kmeans.predict(user_centroid)
    # Add cluster labels to the centroids DataFrame
    centroids_df['cluster'] = kmeans.fit_predict(scaled_centroids_df)

    # ------------------------------ GET USER CLUSTER TRACKS ------------------------------
    # Find rows in centroids_df that have the same cluster as the user
    factors = ['Unnamed: 0', 'artist_name', 'track_name', 'track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']
    relevant = centroids_df[centroids_df['cluster'] == float(user_cluster[0])] # DataFrame of rows with centroids and cluster
    # Get indices of rows
    indices = relevant.index
    relevant_tracks = []
    with open('playlist_data.json', "r") as playlist_file:
        contents = json.load(playlist_file)
        for index in indices: # For each cluster-relevant playlist
            relevant_tracks.extend(contents[index]['Tracks']) # Add its tracks to relevant_tracks
    # ------------------------------ RECOMMENDATIONS ------------------------------
    relevant_tracks = pd.DataFrame(relevant_tracks)
    relevant_features = relevant_tracks[features] # Remove track_name, artist_name, id, ...

    distances = euclidean_distances(user_centroid, relevant_features).flatten()
    
    relevant_tracks['distance'] = distances # Add distances to relevant_tracks DataFrame

    # Sort by Euclidean distance and select top n_recs
    recommendations = relevant_tracks.sort_values(by=['distance'], ascending=[True]).drop_duplicates(subset=['track_id']).head(n_recs)
    return [user_centroid, recommendations]



def calculate_clustering_metrics(k=5):
    features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
    centroids_df = pd.read_csv('centroids_with_clusters.csv')
    # Drop the 'cluster' column from centroids_df to exclude it from the input
    if 'cluster' in centroids_df.columns:
        centroids_df = centroids_df.drop(columns=['cluster'])
    kmeans = KMeans(n_clusters=k)
    # Standardize Feature Columns
    scaler = StandardScaler()
    scaled_centroids_df = scaler.fit_transform(centroids_df[features])
    # KMeans Cluster
    kmeans.fit(scaled_centroids_df)
    cluster_labels = kmeans.labels_
    # Calculate Silhouette Score
    silhouette = silhouette_score(scaled_centroids_df, cluster_labels)
    
    # Calculate Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(scaled_centroids_df, cluster_labels)
    
    # Calculate Calinski-Harabasz Index
    calinski_harabasz = calinski_harabasz_score(scaled_centroids_df, cluster_labels)

    return [silhouette, davies_bouldin, calinski_harabasz]



# driver code
metrics = calculate_clustering_metrics()
print(metrics)

features = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
pl1 = pd.read_csv('../Playlists/hedgehogs_dilemma.csv')
pl2 = pd.read_csv('../Playlists/hers.csv')
pl3 = pd.read_csv('../Playlists/you_wanna_rock.csv')

users = [pl1, pl2, pl3]
results = []
for i in range(3): 
    user_centroid, recs = recommendSongs(users[i])
    user_centroid = user_centroid[features]
    recs = recs[features]
    centroid_vec = recs.mean(axis=0)
    pl_centroid = pd.DataFrame(centroid_vec).transpose()
    print(f"Playlist {i}")
    print(user_centroid)
    print(pl_centroid)
    results.append(cosine_similarity(user_centroid, pl_centroid))

print(results)
print(sum(results)/3) # small sample size of 3, larger sample size in helper.ipynb