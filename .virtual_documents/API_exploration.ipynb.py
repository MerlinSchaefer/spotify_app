import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd


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

sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


token = util.prompt_for_user_token(username1, scope, client_id, client_secret, redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username1)


results_fav_songs = sp.current_user_top_tracks(limit = 50,offset=0,time_range='short_term')


from spotifuncs import create_df_top_songs, append_audio_features


# # Convert it to Dataframe
# track_name = []
# track_id = []
# artist = []
# album = []
# duration = []
# popularity = []
# for i, items in enumerate(results['items']):
#         track_name.append(items['name'])
#         track_id.append(items['id'])
#         artist.append(items["artists"][0]["name"])
#         duration.append(items["duration_ms"])
#         album.append(items["album"]["name"])
#         popularity.append(items["popularity"])

# # Create the final df   
# df_favourite = pd.DataFrame({ "track_name": track_name, 
#                              "album": album, 
#                              "track_id": track_id,
#                              "artist": artist, 
#                              "duration": duration, 
#                              "popularity": popularity})

# df_favourite


df_favourite = create_df_top_songs(results_fav_songs)


df_favourite.shape


df_favourite


df_favourite.to_csv("favourites_0612.csv", encoding="utf-8") #save for later use


df_favourite = append_audio_features(df_favourite,sp)


df_favourite


from pandas_profiling import ProfileReport
prof = ProfileReport(df_favourite)
#prof.to_file(output_file='output.html')


prof


#get cosine similarity for all songs within the playlist get songs that are similar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


df_favourite.head()


features = list(df_favourite.columns[6:])
df_features_favourite = df_favourite[features]
df_features_favourite


df_features_scaled = StandardScaler().fit_transform(df_features_favourite)


cosine_sim = linear_kernel(df_features_scaled, df_features_scaled)
cosine_sim


indices = pd.Series(df_favourite.index, index = df_favourite['track_name']).drop_duplicates()
indices


def get_recommendations(song_title, similarity_score = cosine_sim):
    idx = indices[song_title]
    sim_scores = list(enumerate(similarity_score[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1],reverse = True)
    top_scores = sim_scores[0:6]
    song_indices = [i[0] for i in top_scores]
    return df_favourite["track_name"].iloc[song_indices]


get_recommendations("Evolution")


get_recommendations("Herr Mannelig")


get_recommendations("Lambo Lambo")


for title in df_favourite["track_name"]:
    print(title, "\n")
    print(get_recommendations(title))
    print("------")


from scipy.spatial.distance import cdist

euclid_dist = cdist(df_features_scaled, df_features_scaled, 'euclid')


get_recommendations("Herr Mannelig"), get_recommendations("Herr Mannelig", similarity_score = euclid_dist)



get_recommendations("MOSKAU"), get_recommendations("MOSKAU", similarity_score = euclid_dist)


get_recommendations("Evolution"), get_recommendations("Evolution", similarity_score = euclid_dist)


real_cosine = cosine_similarity(df_features_scaled,df_features_scaled)
real_cosine


get_recommendations("Herr Mannelig"), get_recommendations("Herr Mannelig", similarity_score = real_cosine)


for title in df_favourite["track_name"]:
    print(title, "\n")
    print(get_recommendations(title), "\n \n", get_recommendations(title, similarity_score = real_cosine))
    print("------")
