from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import boto3
from io import BytesIO
import knntools

app = Flask(__name__)
CORS(app)

# s3
s3 = boto3.client('s3')

# load csv
def load_csv_from_s3(bucket_name, key):
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        return df
    except Exception as e:
        print(f"Error loading file from S3: {e}")
        return None
bucket_name = 'spotifydatacsv'
file_key = 'spotify_data.csv'
tracks = load_csv_from_s3(bucket_name, file_key)

if tracks is not None:
    # load and preprocess data
    tracks['artist_name'] = tracks['artist_name'].str.strip().str.lower()
    tracks['track_name'] = tracks['track_name'].str.strip().str.lower()
    tracks['genre'] = tracks['genre'].str.strip().str.lower()
    tracks.drop_duplicates(inplace=True)
    tracks.dropna(inplace=True)
    tracks = pd.get_dummies(tracks, columns=['genre'])

    # encode artist and track names
    le_artist = LabelEncoder()
    le_track = LabelEncoder()
    tracks['artist_name'] = le_artist.fit_transform(tracks['artist_name'])
    tracks['track_name'] = le_track.fit_transform(tracks['track_name'])

    features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_ms', 'time_signature']

    kmeans_feature_columns = [col for col in features if col in tracks.columns]
    feature_columns = [col for col in tracks.columns if col not in ['track_id', 'Unnamed: 0', 'cluster']]
    tracks = tracks.drop(columns=['track_id', 'Unnamed: 0'])

    scaler_kmeans = StandardScaler()
    kmeans_scaled_features = scaler_kmeans.fit_transform(tracks[kmeans_feature_columns])

    scaler_knn = StandardScaler()
    scaled_features = scaler_knn.fit_transform(tracks[feature_columns])

    knn = NearestNeighbors(n_neighbors=11)
    knn.fit(scaled_features)

    kmeans = KMeans(n_clusters=10, random_state=42)
    tracks['cluster'] = kmeans.fit_predict(kmeans_scaled_features)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    print(f"Received data: {data}")
    song_name = data['songName'].strip().lower()
    artist_name = data['artistName'].strip().lower()
    model = data.get('model', 'knn').strip().lower()

    print(f"Received request for song: {song_name} by artist: {artist_name} using model: {model}")

    if tracks is None:
        return jsonify({"error": "Failed to load data"}), 500

    if model == 'kmeans':
        print("Using KMeans model")
        recommendations = recommend_songs_kmeans(song_name, artist_name, kmeans_feature_columns, scaler_kmeans, kmeans, tracks, le_artist, le_track)
    elif model == 'knn':
        print("Using KNN model")
        input_song = knntools.find_song(artist_name, song_name, tracks, le_artist, le_track)
        if isinstance(input_song, str):
            print("Song not found for KNN")
            return jsonify([])
        input_song_dict = input_song.iloc[0].to_dict()
        similar_songs = knntools.find_similar_songs(input_song_dict, tracks, knn, scaler_knn, le_artist, le_track)
        recommendations = similar_songs.to_dict(orient='records')[1:]
    else:
        print("Invalid model specified")
        return jsonify({"error": "Invalid model specified"}), 400

    print(f"Recommendations generated using {model} model")
    return jsonify(recommendations)

def recommend_songs_kmeans(inputTrackName, inputArtistName, feature_columns, scaler, kmeans, tracks, le_artist, le_track, n_recommendations=10):
    inputTrackName = inputTrackName.strip().lower()
    inputArtistName = inputArtistName.strip().lower()

    print(f"Encoding artist: {inputArtistName} and track: {inputTrackName}")

    try:
        artist_name_encoded = le_artist.transform([inputArtistName])[0]
        track_name_encoded = le_track.transform([inputTrackName])[0]
    except ValueError:
        print("Track or artist not found in encoding")
        return []

    print(f"Encoded artist: {artist_name_encoded}, track: {track_name_encoded}")

    inputTrack = tracks[(tracks['track_name'] == track_name_encoded) & (tracks['artist_name'] == artist_name_encoded)]

    if inputTrack.empty:
        print("Track not found in dataset")
        return []

    print("Input track found:")
    print(inputTrack)

    inputTrackFeatures = inputTrack[feature_columns]
    scaledInputTrackFeatures = scaler.transform(inputTrackFeatures)

    # predict cluster
    inputCluster = kmeans.predict(scaledInputTrackFeatures)[0]
    print(f"Predicted cluster: {inputCluster}")

    # get tracks from the same cluster
    similarTracks = tracks[tracks['cluster'] == inputCluster]
    similarTracks = similarTracks[(similarTracks['track_name'] != track_name_encoded) | (similarTracks['artist_name'] != artist_name_encoded)]

    similarTracksFeatures = scaler.transform(similarTracks[feature_columns])
    distances = euclidean_distances(scaledInputTrackFeatures, similarTracksFeatures).flatten()
    similarTracks['distance'] = distances

    # sort by distance and select top 10
    recommendations = similarTracks.sort_values(by=['distance', 'popularity'], ascending=[True, False]).drop_duplicates(subset=['artist_name', 'track_name']).head(n_recommendations)
    if len(recommendations) < n_recommendations:
        remaining = similarTracks[~similarTracks.index.isin(recommendations.index)].head(n_recommendations - len(recommendations))
        recommendations = pd.concat([recommendations, remaining])

    recommendations['artist_name'] = le_artist.inverse_transform(recommendations['artist_name'])
    recommendations['track_name'] = le_track.inverse_transform(recommendations['track_name'])

    print("Recommendations:")
    print(recommendations)

    columns_to_return = ['artist_name', 'track_name', 'popularity', 'year']
    if 'genre' in tracks.columns:
        columns_to_return.append('genre')

    return recommendations[columns_to_return].to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
