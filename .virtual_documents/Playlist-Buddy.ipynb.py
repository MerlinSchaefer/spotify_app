import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd
import numpy as np
from spotifuncs import *


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


scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-modify-private playlist-read-private playlist-read-collaborative"

redirect_uri = "https://developer.spotify.com/dashboard/applications/4a4e029d299a4241873db8300038bf0a"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                      client_secret=client_secret)



sp_m = authenticate(redirect_uri, client_credentials_manager, username1, scope, client_id, client_secret)


redlight = sp_m.playlist("spotify:playlist:3zcSUFp0puoWXWXuFCF2e6")


redlight_df = create_df_playlist(redlight) #see if assert error works


redlight_df = create_df_playlist(redlight, sp = sp_m)
redlight_df


cols = redlight_df.columns[6:].tolist()
cols


redlight_df[cols].mean()


mean_song = pd.DataFrame(columns=redlight_df.columns)
mean_song.loc["mean"] = redlight_df.mean()


mean_song


seed_tracks = redlight_df["track_id"].tolist()


len(seed_tracks)


recomm_dfs = []
for i in range(5,len(seed_tracks)+1,5):
    recomms = sp_m.recommendations(seed_tracks = seed_tracks[i-5:i],limit = 25)
    recomms_df = append_audio_features(create_df_recommendations(recomms),sp_m)
    recomm_dfs.append(recomms_df)
recomms_df = pd.concat(recomm_dfs)
recomms_df.reset_index(drop = True, inplace = True)


recomms_df


##testing whether MinMax Scaling works better than StandardScaling
## MinMax has been adopted for the spotifuncs function
# from sklearn.preprocessing import MinMaxScaler
# def test_create_similarity_score(df1,df2,similarity_score = "cosine_sim"):
#     """ 
#     Creates a similarity matrix for the audio features (except key and mode) of two Dataframes.

#     Parameters
#     ----------
#     df1 : DataFrame containing track_name,track_id, artist,album,duration,popularity
#             and all audio features
#     df2 : DataFrame containing track_name,track_id, artist,album,duration,popularity
#             and all audio features
    
#     similarity_score: similarity measure (linear,cosine_sim)

#     Returns
#     -------
#     A matrix of similarity scores for the audio features of both DataFrames.
#     """
    
#     assert list(df1.columns[6:]) == list(df2.columns[6:]), "dataframes need to contain the same columns"
#     features = list(df1.columns[6:])
#     features.remove('key')
#     features.remove('mode')
#     df_features1,df_features2 = df1[features],df2[features]
#     scaler = MinMaxScaler()
#     df_features_scaled1,df_features_scaled2 = scaler.fit_transform(df_features1),scaler.fit_transform(df_features2)
#     if similarity_score == "linear":
#         linear_sim = linear_kernel(df_features_scaled1, df_features_scaled2)
#         return linear_sim
#     elif similarity_score == "cosine_sim":
#         cosine_sim = cosine_similarity(df_features_scaled1, df_features_scaled2)
#         return cosine_sim
#     #other measures may be implemented in the future



similarity_score = create_similarity_score(redlight_df,recomms_df)



#test_similarity_score = test_create_similarity_score(redlight_df,recomms_df)


similarity_score.shape#, test_similarity_score.shape


[np.argmax(i) for i in similarity_score]#, [np.argmax(i) for i in test_similarity_score]


final_recomms = recomms_df.iloc[[np.argmax(i) for i in similarity_score]]
final_recomms = final_recomms.drop_duplicates().reset_index(drop = True)


#test_final_recomms = recomms_df.iloc[[np.argmax(i) for i in test_similarity_score]]
#test_final_recomms = test_final_recomms.drop_duplicates().reset_index(drop = True)


final_recomms


#test_final_recomms


len(final_recomms)


final_recomms[final_recomms["track_name"].isin(redlight_df["track_name"])]


#filter again so tracks are not already in playlist_df
final_recomms = final_recomms[~final_recomms["track_name"].isin(redlight_df["track_name"])]
final_recomms.reset_index(drop = True, inplace = True)


#filter again so tracks are not already in playlist_df
#test_final_recomms = test_final_recomms[~test_final_recomms["track_name"].isin(redlight_df["track_name"])]
#test_final_recomms.reset_index(drop = True, inplace = True)


final_recomms


#test_final_recomms


#add both and compare wait between to
# sp_m.user_playlist_add_tracks(usernames[0],
#                              playlist_id="spotify:playlist:36MtjIS6lPXT7Q97HieR9g",
#                              tracks = final_recomms["track_id"].tolist())


# sp_m.user_playlist_add_tracks(usernames[0],
#                              playlist_id="spotify:playlist:1ypSXCaY044CeRD98pteRc",
#                              tracks = test_final_recomms["track_id"].tolist())


##adopted into spotifuncs
# def test_filter_with_meansong(mean_song,recommendations_df, n_recommendations = 10):
#     features = list(mean_song.columns[6:])
#     features.remove("key")
#     features.remove("mode")
#     mean_song_feat = mean_song[features].values
#     mean_song_scaled = MinMaxScaler().fit_transform(mean_song_feat.reshape(-1,1))
#     recommendations_df_scaled = MinMaxScaler().fit_transform(recommendations_df[features])
#     mean_song_scaled = mean_song_scaled.reshape(1,-1)
#     sim_mean_finrecomms = cosine_similarity(mean_song_scaled,recommendations_df_scaled)[0][:]
#     #sim_mean_finrecomms = sim_mean_finrecomms[0][:]
#     indices = (-sim_mean_finrecomms).argsort()[:n_recommendations]
#     final_recommendations = recommendations_df.iloc[indices]
#     return final_recommendations


#manual filtering
final_recomms.columns


final_recomms.describe()


##function in spotifuncs
# def feature_filter(df,feature, high = True):
#     assert feature in ["speechiness","acousticness",
#                        "instrumentalness","liveness"], "feature must be one of the following: speechiness,acousticness,instrumentalness,liveness"
#     x = 0.9 if high == True else 0.1
#     df = df[df[feature] > x] if high == True else df[df[feature] < x]
#     return df


feature_filter(final_recomms,feature = "speechiness")
        


feature_filter(final_recomms,feature = "speechiness", high = False)


feature_filter(final_recomms,feature = "acousticness")


feature_filter(final_recomms,feature = "acousticness", high = False)


feature_filter(final_recomms,feature = "instrumentalness")


feature_filter(final_recomms,feature = "instrumentalness", high = False)


feature_filter(final_recomms,feature = "liveness")


feature_filter(final_recomms,feature = "liveness", high = False)


final_recomms = filter_with_meansong(mean_song,final_recomms)


#testScaling_final_recomms = test_filter_with_meansong(mean_song,test_final_recomms)


#testScaling_final_recomms


#test_final_recomms[test_final_recomms["track_name"] == "Pandora"]


final_recomms


sp_m.user_playlist_add_tracks(usernames[0],
                              playlist_id="spotify:playlist:4XP9wRGPImiYYiGtsB6Dd3",
                              tracks = final_recomms["track_id"].tolist())



