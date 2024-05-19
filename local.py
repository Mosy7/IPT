import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging
import webbrowser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Spotify API credentials
SPOTIPY_CLIENT_ID = '619ba539d495469499461b6df2261d29'
SPOTIPY_CLIENT_SECRET = 'a0979d56c7274136a6a0b22c830794a1'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# Define the scope for the permissions you need
scope = "user-library-read playlist-modify-public playlist-modify-private"

def authenticate_spotify():
    """
    Authenticate with Spotify and return the Spotify client object.
    """
    sp_oauth = SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                            client_secret=SPOTIPY_CLIENT_SECRET,
                            redirect_uri=SPOTIPY_REDIRECT_URI,
                            scope=scope)
    
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        print("Please navigate to the following URL to authenticate:")
        print(auth_url)
        
        # Open the URL in the default web browser
        webbrowser.open(auth_url)
        
        # Wait for user to authenticate and provide the redirected URL
        response = input("Enter the URL you were redirected to: ")
        code = sp_oauth.parse_response_code(response)
        token_info = sp_oauth.get_access_token(code)
    
    sp = spotipy.Spotify(auth=token_info['access_token'])
    return sp

def collect_data(sp):
    """
    Collect data from Spotify API.
    """
    logging.info("Starting data collection")
    try:
        results = sp.current_user_saved_tracks(limit=50)
        tracks = results['items']

        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])

        data = []
        for item in tracks:
            track = item['track']
            data.append({
                'id': track['id'],
                'name': track['name'],
                'popularity': track['popularity'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'release_date': track['album']['release_date']
            })

        logging.info("Data collection complete with {} tracks".format(len(data)))
        return pd.DataFrame(data)
    except Exception as e:
        logging.error("Error during data collection: {}".format(e))
        return pd.DataFrame()

def preprocess_data(data):
    """
    Preprocess the collected data.
    """
    logging.info("Starting data preprocessing")
    try:
        data = data.dropna()
        data['popularity'] = data['popularity'] / 100  # Normalize popularity to range [0, 1]
        logging.info("Data preprocessing complete")
        return data
    except Exception as e:
        logging.error("Error during data preprocessing: {}".format(e))
        return pd.DataFrame()

def build_recommendation_model(data):
    """
    Build and train the recommendation model.
    """
    logging.info("Starting model training")
    try:
        X = data[['popularity']]  # Features (you can add more features)
        y = data['id']  # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        
        logging.info("Model training complete")
        return model
    except Exception as e:
        logging.error("Error during model training: {}".format(e))
        return None

def generate_recommendations(model, user_data):
    """
    Generate track recommendations for a user.
    """
    logging.info("Generating recommendations")
    try:
        recommendations = model.predict(user_data)
        logging.info("Recommendations generated")
        return recommendations.tolist()
    except Exception as e:
        logging.error("Error during recommendation generation: {}".format(e))
        return []

def get_spotify_track_ids(sp, track_names):
    """
    Convert a list of track names to Spotify track IDs.
    """
    logging.info("Converting track names to Spotify track IDs")
    track_ids = []
    for name in track_names:
        track_id = search_track(sp, name)
        if track_id:
            track_ids.append(track_id)
    logging.info("Conversion complete with {} track IDs".format(len(track_ids)))
    return track_ids

def search_track(sp, track_name):
    """
    Search for a track by name and return its Spotify ID.
    """
    results = sp.search(q=track_name, limit=1)
    tracks = results['tracks']['items']
    if tracks:
        return tracks[0]['id']
    return None

def create_playlist(sp, user_id, playlist_name, track_ids):
    """
    Create a new playlist for the user and add tracks to it.
    """
    logging.info("Creating playlist")
    try:
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True)
        sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist['id'], tracks=track_ids)
        logging.info("Playlist created with ID: {}".format(playlist['id']))
        return playlist['id']
    except Exception as e:
        logging.error("Error during playlist creation: {}".format(e))
        return None

def main():
    # Authenticate with Spotify
    sp = authenticate_spotify()
    
    # Collect data
    data = collect_data(sp)

    if data.empty:
        logging.error("No data collected, terminating program.")
        return

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    if preprocessed_data.empty:
        logging.error("Data preprocessing failed, terminating program.")
        return

    # Build recommendation model
    model = build_recommendation_model(preprocessed_data)

    if not model:
        logging.error("Model training failed, terminating program.")
        return

    # Get user data for recommendations (this is a placeholder)
    # Implement logic to get user data
    user_data = np.array([[0.7]])  # Replace with actual user data

    # Generate recommendations
    recommended_tracks = generate_recommendations(model, user_data)

    if not recommended_tracks:
        logging.error("No recommendations generated, terminating program.")
        return

    # Get Spotify track IDs for the recommended tracks
    spotify_track_ids = get_spotify_track_ids(sp, recommended_tracks)

    if not spotify_track_ids:
        logging.error("No Spotify track IDs found, terminating program.")
        return

    # Get the current user ID
    user_id = sp.me()['id']
    logging.info("Authenticated as user: {}".format(user_id))

    # Create a new playlist and add the recommended tracks
    playlist_name = "My Recommended Playlist"
    playlist_id = create_playlist(sp, user_id, playlist_name, spotify_track_ids)

    if not playlist_id:
        logging.error("Playlist creation failed, terminating program.")
    else:
        logging.info("Program completed successfully, playlist created.")

if __name__ == '__main__':
    main()
