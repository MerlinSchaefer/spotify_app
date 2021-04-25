##script for duos playlist

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd
import numpy as np
from spotifuncs import *
from collections import Counter
from operator import itemgetter
from heapq import nlargest

##define function to get required dfs, subject to change, not part of the spotifuncs
def get_dfs(sp):
    ##queries
    #user top tracks
    top_tracks_short = sp.current_user_top_tracks(limit = 50,offset=0,time_range='short_term')
    top_tracks_med = sp.current_user_top_tracks(limit = 50,offset=0,time_range='medium_term')
    top_tracks_long = sp.current_user_top_tracks(limit = 50,offset=0,time_range='long_term')
    
    #combine the top_tracks
    top_tracks_short_df = append_audio_features(create_df_top_songs(top_tracks_short),sp)
    top_tracks_med_df = append_audio_features(create_df_top_songs(top_tracks_med),sp)
    top_tracks_long_df = append_audio_features(create_df_top_songs(top_tracks_long),sp)
    #sample from long-term top tracks to introduce more randomness and avoid having the same artists
    top_tracks_long_df = top_tracks_long_df.sample(n = 15)
    top_tracks_df = pd.concat([top_tracks_short_df,top_tracks_med_df,top_tracks_long_df]).drop_duplicates().reset_index(drop = True)
        
    #user top artists
    top_artists_long = sp.current_user_top_artists(limit = 50, time_range = "long_term")
    top_artists_med = sp.current_user_top_artists(limit = 50, time_range = "medium_term")
    top_artists_short = sp.current_user_top_artists(limit = 50, time_range = "short_term")
    
    artists_short_df = top_artists_from_API(top_artists_short)
    artists_med_df = top_artists_from_API(top_artists_med)
    artists_long_df = top_artists_from_API(top_artists_long)
    artists_df = pd.concat([artists_short_df,artists_med_df,artists_long_df])
    artists_df["genres"] = artists_df["genres"].apply(lambda x: ",".join(x))
    artists_df.drop_duplicates().reset_index(drop = True)
    
    #user saved tracks
    user_saved_tracks = sp.current_user_saved_tracks(limit = 50)
    saved_tracks_df = create_df_saved_songs(user_saved_tracks)
    
        
    return top_tracks_df,artists_df,saved_tracks_df



#set path
path = Path("C:/Users/ms101/OneDrive/DataScience_ML/projects/spotify_app")
#Path("/home/merlin/OneDrive/DataScience_ML/projects/spotify_app")#
#get client_id and client_secret
with open(path / "client_s.txt") as f:
    content = f.readlines()
content = [x.strip() for x in content]

client_id = content[0]
client_secret = content[1]
#get usernames
with open(path / "usernames.txt") as f:
    usernames = f.readlines()
usernames = [x.strip() for x in usernames]

username1 = usernames[0]
username2 = usernames[1]
#set scope, uri and client_credentials
scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-read-private playlist-read-collaborative playlist-modify-private"

redirect_uri = "https://developer.spotify.com/dashboard/applications/4a4e029d299a4241873db8300038bf0a"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                      client_secret=client_secret)

#authenticate both users
sp_m = authenticate(redirect_uri, client_credentials_manager, username1, scope, client_id, client_secret)
sp_t = authenticate(redirect_uri, client_credentials_manager, username2, scope, client_id, client_secret)

# get dfs for both users
top_tracks_m, artists_m, saved_tracks_m = get_dfs(sp_m)
top_tracks_t, artists_t, saved_tracks_t = get_dfs(sp_t)

#load last weeks playlist
last_week_duo = pd.read_csv(path/"Playlist.csv", index_col = 0)

#find common artists
common_artists = dataframe_difference(artists_m,artists_t, which = "both")
common_artists.to_csv(path / "common_artists.csv") #save for checking

#initiate new playlist dataframe with common top songs that were not in last weeks playlist
common_songs = dataframe_difference(top_tracks_m,top_tracks_t,which = "both")
new_playlist_df = common_songs[~common_songs["track_id"].isin(last_week_duo["track_id"])]

#Create a similarity matrix for top songs of both users, delete songs that both dataframes contain first.
#create dfs of unique songs for each person
unique_top_tracks_m = top_tracks_m[~top_tracks_m["track_id"].isin(common_songs["track_id"])]
unique_top_tracks_m.reset_index(drop = True,inplace = True)
unique_top_tracks_t = top_tracks_t[~top_tracks_t["track_id"].isin(common_songs["track_id"])]
unique_top_tracks_t.reset_index(drop = True,inplace = True)
#get 30 most similar songs
similarity_top_songs = create_similarity_score(unique_top_tracks_m,unique_top_tracks_t)
max_n_scores = [(i,np.argmax(x),x[np.argmax(x)]) for i,x in enumerate(similarity_top_songs)]
idx_simtracks_m = [i[0] for i in  nlargest(30,max_n_scores,key=itemgetter(2))]
idx_simtracks_t = [i[1] for i in  nlargest(30,max_n_scores,key=itemgetter(2))]
sim_top_tracks_m = unique_top_tracks_m.loc[idx_simtracks_m]
sim_top_tracks_t = unique_top_tracks_t.loc[idx_simtracks_t]
similar_top_tracks = pd.concat([sim_top_tracks_m,sim_top_tracks_t])
similar_top_tracks.drop_duplicates(inplace = True)
similar_top_tracks = similar_top_tracks[~similar_top_tracks["track_id"].isin(last_week_duo["track_id"])]
similar_top_tracks.reset_index(drop = True,inplace = True)
#append 10 sampled songs from most similar songs
new_playlist_df = new_playlist_df.append(similar_top_tracks.sample(10))
#filter top tracks with common artists for both users
filtered_top_m = top_tracks_m[top_tracks_m["artist"].isin(common_artists["name"]) 
                              & ~top_tracks_m["track_id"].isin(last_week_duo["track_id"])]
filtered_top_m.to_csv(path / "filtered_top_m.csv") #save for checking
filtered_top_t = top_tracks_t[top_tracks_t["artist"].isin(common_artists["name"])
                             & ~top_tracks_t["track_id"].isin(last_week_duo["track_id"])]
filtered_top_t.to_csv(path / "filtered_top_t.csv") #save for checking

#to not have 2+ songs by the same artist we will sample from the above dataframes
#I will assign weights to the rows depending on how often an artist occurs
weights_m = [1/len(filtered_top_m)/7 if Counter(filtered_top_m["artist"])[x] > 2 else 1/len(filtered_top_m) for x in filtered_top_m["artist"]] 
weights_t = [1/len(filtered_top_t)/7 if Counter(filtered_top_t["artist"])[x] > 2 else 1/len(filtered_top_m) for x in filtered_top_t["artist"]] 

#determine sample size
sample_n = (25-len(new_playlist_df))//2
if sample_n > 7: sample_n = 7

#add samples to new_playlist_df
if sample_n > len(filtered_top_m): sample_n = len(filtered_top_m)
new_playlist_df = new_playlist_df.append(filtered_top_m.sample(sample_n,weights = weights_m))
if sample_n > len(filtered_top_t): sample_n = len(filtered_top_t)
new_playlist_df = new_playlist_df.append(filtered_top_t.sample(sample_n,weights = weights_t))
new_playlist_df = new_playlist_df.drop_duplicates().reset_index(drop=True)

#sample the remaining 25-len(new_playlist_df) from saved_tracks
#first get audio_features
saved_tracks_m = append_audio_features(saved_tracks_m, sp_m)
saved_tracks_t = append_audio_features(saved_tracks_t,sp_t)
#filter again so artists are not already in new_playlist_df
filtered_saved_m = saved_tracks_m[~saved_tracks_m["artist"].isin(new_playlist_df["artist"])]
filtered_saved_t = saved_tracks_t[~saved_tracks_t["artist"].isin(new_playlist_df["artist"])]
sample_n = (25-len(new_playlist_df))//2

new_playlist_df = pd.concat([new_playlist_df,filtered_saved_m.sample(sample_n),filtered_saved_t.sample(sample_n)])
new_playlist_df.dropna(inplace = True)
new_playlist_df.reset_index(drop = True, inplace= True)

#get seed_tracks from new_playlist_df for recommendations 
seed_tracks = new_playlist_df["track_id"].tolist()
#create recommendations df
recomm_dfs = []
for i in range(5,26,5):
    recomms = sp_m.recommendations(seed_tracks = seed_tracks[i-5:i],limit = 25)
    recomms_df = append_audio_features(create_df_recommendations(recomms),sp_m)
    recomm_dfs.append(recomms_df)
recomms_df = pd.concat(recomm_dfs)
recomms_df.reset_index(drop = True, inplace= True)

#create similarity matrix between new_playlist_df and recomms_df
similarity_score = create_similarity_score(new_playlist_df,recomms_df)
#get final recommendations through similarity scores
final_recomms = recomms_df.loc[[np.argmax(i) for i in similarity_score]]
final_recomms = final_recomms.drop_duplicates()
#append to new_playlist df
new_playlist_df = new_playlist_df.append(final_recomms)
new_playlist_df = new_playlist_df.drop_duplicates()
new_playlist_df.reset_index(drop = True, inplace = True)
new_playlist_df.to_csv(path / "Playlist.csv")

#view tracks and artists to check if everything worked
print("These are the songs for our new DUO.py playlist")
print(*zip(new_playlist_df["track_name"],new_playlist_df["artist"]))
#add tracks to playlist
confirm = input("Please confirm that you want to replace the current playlist by typing YES   ")
if confirm == "YES":
    sp_m.playlist_replace_items(playlist_id="spotify:playlist:1Vcqtv3nE7QOJ4KFvK7bT8",
                              items = new_playlist_df["track_id"].tolist())