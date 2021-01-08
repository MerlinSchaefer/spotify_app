import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd
from spotifuncs import authenticate,create_df_top_songs, append_audio_features,create_similarity_score,get_recommendations, top_artists_from_API, create_df_saved_songs


path = Path("C:/Users/ms101/OneDrive/DataScience_ML/projects/spotify_app")


with open(path / "client_s.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]

client_id = content[0]
client_secret = content[1]


with open(path / "usernames.txt") as f:
    usernames = f.readlines()
usernames = [x.strip() for x in usernames]

username1 = usernames[0]
username2 = usernames[1]


scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-read-private playlist-read-collaborative"


redirect_uri = "https://developer.spotify.com/dashboard/applications/4a4e029d299a4241873db8300038bf0a"


client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                       client_secret=client_secret)

# sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# token = util.prompt_for_user_token(username2, scope, client_id, client_secret, redirect_uri)

# if token:
#     sp = spotipy.Spotify(auth=token)
# else:
#     print("Can't get token for", username2)


sp = authenticate(redirect_uri, client_credentials_manager, username2, scope, client_id, client_secret)


##queries
#user top tracks
top_tracks_short = sp.current_user_top_tracks(limit = 50,offset=0,time_range='short_term')
top_tracks_med = sp.current_user_top_tracks(limit = 50,offset=0,time_range='medium_term')
top_tracks_long = sp.current_user_top_tracks(limit = 50,offset=0,time_range='long_term')
#user top artists
top_artists_long = sp.current_user_top_artists(limit = 50, time_range = "long_term")
top_artists_med = sp.current_user_top_artists(limit = 50, time_range = "medium_term")
top_artists_short = sp.current_user_top_artists(limit = 50, time_range = "short_term")
#user recent tracks
#still in BETA phase no spotipy function (workaround??)

#user saved tracks
user_saved_tracks = sp.current_user_saved_tracks(limit = 50)



artists_short_df = top_artists_from_API(top_artists_short)
artists_med_df = top_artists_from_API(top_artists_med)
artists_long_df = top_artists_from_API(top_artists_long)
artists_df = pd.concat([artists_short_df,artists_med_df,artists_long_df])
artists_df["genres"] = artists_df["genres"].apply(lambda x: ",".join(x))
artists_df.drop_duplicates().reset_index(drop = True)
artists_df


artists_short_df


artists_med_df


artists_long_df


user_saved_tracks["items"][0]["track"].keys()


user_saved_tracks["items"][0]["track"]["name"]


user_saved_tracks["items"][0]["track"]["album"]["name"]


user_saved_tracks["items"][0]["track"]["artists"][0]["name"]


user_saved_tracks["items"][0]["track"]["popularity"]


user_saved_tracks["items"][0]["track"]["id"]


user_saved_tracks["items"][0]["track"]["duration_ms"]


saved_tracks_df = create_df_saved_songs(user_saved_tracks)


saved_tracks_df


#combine the top_tracks
top_tracks_short_df = append_audio_features(create_df_top_songs(top_tracks_short),sp)
top_tracks_med_df = append_audio_features(create_df_top_songs(top_tracks_med),sp)
top_tracks_long_df = append_audio_features(create_df_top_songs(top_tracks_long),sp)
top_tracks_df = pd.concat([top_tracks_short_df,top_tracks_med_df,top_tracks_long_df]).drop_duplicates().reset_index(drop = True)
top_tracks_df


top_tracks_short_df


top_tracks_med_df


top_tracks_long_df


saved_tracks_df = append_audio_features(saved_tracks_df,sp)


saved_tracks_df.head(10)


similarity_top_tracks = create_similarity_score(top_tracks_df,top_tracks_df)
similarity_saved_tracks = create_similarity_score(saved_tracks_df,saved_tracks_df)


for track in top_tracks_df["track_name"].sample(5):
    print(track, "\n")
    print(get_recommendations(top_tracks_df,track,similarity_top_tracks))
    print("\n")


for track in saved_tracks_df["track_name"].sample(5):
    print(track, "\n")
    print(get_recommendations(saved_tracks_df,track,similarity_saved_tracks))
    print("\n")


##Playlists
#spotify features playlists + get playlist by id
#featured_playlists, playlist
