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
    top_tracks_1ed = sp.current_user_top_tracks(limit = 50,offset=0,time_range='medium_term')
    top_tracks_long = sp.current_user_top_tracks(limit = 50,offset=0,time_range='long_term')
    
    #combine the top_tracks
    top_tracks_short_df = append_audio_features(create_df_top_songs(top_tracks_short),sp)
    top_tracks_1ed_df = append_audio_features(create_df_top_songs(top_tracks_1ed),sp)
    top_tracks_long_df = append_audio_features(create_df_top_songs(top_tracks_long),sp)
    #sample from long-term top tracks to introduce more randomness and avoid having the same artists
    top_tracks_long_df = top_tracks_long_df.sample(n = 15)
    top_tracks_df = pd.concat([top_tracks_short_df,top_tracks_1ed_df,top_tracks_long_df]).drop_duplicates().reset_index(drop = True)
        
    #user top artists
    top_artists_long = sp.current_user_top_artists(limit = 50, time_range = "long_term")
    top_artists_1ed = sp.current_user_top_artists(limit = 50, time_range = "medium_term")
    top_artists_short = sp.current_user_top_artists(limit = 50, time_range = "short_term")
    
    artists_short_df = top_artists_from_API(top_artists_short)
    artists_1ed_df = top_artists_from_API(top_artists_1ed)
    artists_long_df = top_artists_from_API(top_artists_long)
    artists_df = pd.concat([artists_short_df,artists_1ed_df,artists_long_df])
    artists_df["genres"] = artists_df["genres"].apply(lambda x: ",".join(x))
    artists_df.drop_duplicates().reset_index(drop = True)
    
    #user saved tracks
    user_saved_tracks = sp.current_user_saved_tracks(limit = 50)
    saved_tracks_df = create_df_saved_songs(user_saved_tracks)
    
        
    return top_tracks_df,artists_df,saved_tracks_df



#set path
path = Path("-YOUR PATH (SHOULD CONTAIN spotifuncs.py)-")
#get client credentials and user name

#### YOU CAN CREATE A .txt with the credentials and read them with the following code if you don't want to enter them directly####
# with open(path / "client_s.txt") as f:
#     content = f.readlines()
# content = [x.strip() for x in content]
# client_id = content[0]
# client_secret = content[1]
#get usernames
client_id = "-YOUR CLIENT_ID-"
client_secret = "-YOUR CLIENT_SECRET-"
#### YOU CAN CREATE A .txt with the username(s) and read them with the following code if you don't want to enter them directly####
# with open(path / "usernames.txt") as f:
#     usernames = f.readlines()
# usernames = [x.strip() for x in usernames]
# username1 = usernames[0]
#username2 = usernames[1]

username1 = "-YOUR USERNAME-"
username2 = "-YOUR USERNAME #2"

#set scope, uri and client_credentials
scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-read-private playlist-read-collaborative"#can be reduced depending on your playlists

redirect_uri = "-YOUR REDIRECT_URI FROM YOUR SPOTIFY DEVELOPER DASHBOARD-"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                      client_secret=client_secret)

#authenticate both users
sp_1 = authenticate(redirect_uri, client_credentials_manager, username1, scope, client_id, client_secret)
sp_2 = authenticate(redirect_uri, client_credentials_manager, username2, scope, client_id, client_secret)

# get dfs for both users
top_tracks_1, artists_1, saved_tracks_1 = get_dfs(sp_1)
top_tracks_2, artists_2, saved_tracks_2 = get_dfs(sp_2)

#load last weeks playlist
last_week_duo = pd.read_csv(path/"Playlist.csv", index_col = 0) #create an empty csv before running the first time # or fill it with songs you don't wish to see in the first DOU.py

#find common artists
common_artists = dataframe_difference(artists_1,artists_2, which = "both")
common_artists.to_csv(path / "common_artists.csv") #save for checking

#initiate new playlist dataframe with common top songs that were not in last weeks playlist
common_songs = dataframe_difference(top_tracks_1,top_tracks_2,which = "both")
new_playlist_df = common_songs[~common_songs["track_id"].isin(last_week_duo["track_id"])]

#Create a similarity matrix for top songs of both users, delete songs that both dataframes contain first.
#create dfs of unique songs for each person
unique_top_tracks_1 = top_tracks_1[~top_tracks_1["track_id"].isin(common_songs["track_id"])]
unique_top_tracks_1.reset_index(drop = True,inplace = True)
unique_top_tracks_2 = top_tracks_2[~top_tracks_2["track_id"].isin(common_songs["track_id"])]
unique_top_tracks_2.reset_index(drop = True,inplace = True)
#get 30 most similar songs
similarity_top_songs = create_similarity_score(unique_top_tracks_1,unique_top_tracks_2)
max_n_scores = [(i,np.argmax(x),x[np.argmax(x)]) for i,x in enumerate(similarity_top_songs)]
idx_simtracks_1 = [i[0] for i in  nlargest(30,max_n_scores,key=itemgetter(2))]
idx_simtracks_2 = [i[1] for i in  nlargest(30,max_n_scores,key=itemgetter(2))]
sim_top_tracks_1 = unique_top_tracks_1.loc[idx_simtracks_1]
sim_top_tracks_2 = unique_top_tracks_2.loc[idx_simtracks_2]
similar_top_tracks = pd.concat([sim_top_tracks_1,sim_top_tracks_2])
similar_top_tracks.drop_duplicates(inplace = True)
similar_top_tracks = similar_top_tracks[~similar_top_tracks["track_id"].isin(last_week_duo["track_id"])]
similar_top_tracks.reset_index(drop = True,inplace = True)
#append 10 sampled songs from most similar songs
new_playlist_df = new_playlist_df.append(similar_top_tracks.sample(10))
#filter top tracks with common artists for both users
filtered_top_1 = top_tracks_1[top_tracks_1["artist"].isin(common_artists["name"]) 
                              & ~top_tracks_1["track_id"].isin(last_week_duo["track_id"])]
filtered_top_1.to_csv(path / "filtered_top_1.csv") #save for checking
filtered_top_2 = top_tracks_2[top_tracks_2["artist"].isin(common_artists["name"])
                             & ~top_tracks_2["track_id"].isin(last_week_duo["track_id"])]
filtered_top_2.to_csv(path / "filtered_top_2.csv") #save for checking

#to not have 2+ songs by the same artist we will sample from the above dataframes
#I will assign weights to the rows depending on how often an artist occurs
weights_1 = [1/len(filtered_top_1)/7 if Counter(filtered_top_1["artist"])[x] > 2 else 1/len(filtered_top_1) for x in filtered_top_1["artist"]] 
weights_2 = [1/len(filtered_top_2)/7 if Counter(filtered_top_2["artist"])[x] > 2 else 1/len(filtered_top_1) for x in filtered_top_2["artist"]] 

#determine sample size
sample_n = (25-len(new_playlist_df))//2
if sample_n > 7: sample_n = 7

#add samples to new_playlist_df
if sample_n > len(filtered_top_1): sample_n = len(filtered_top_1)
new_playlist_df = new_playlist_df.append(filtered_top_1.sample(sample_n,weights = weights_1))
if sample_n > len(filtered_top_2): sample_n = len(filtered_top_2)
new_playlist_df = new_playlist_df.append(filtered_top_2.sample(sample_n,weights = weights_2))
new_playlist_df = new_playlist_df.drop_duplicates().reset_index(drop=True)

#sample the remaining 25-len(new_playlist_df) from saved_tracks
#first get audio_features
saved_tracks_1 = append_audio_features(saved_tracks_1, sp_1)
saved_tracks_2 = append_audio_features(saved_tracks_2,sp_2)
#filter again so artists are not already in new_playlist_df
filtered_saved_1 = saved_tracks_1[~saved_tracks_1["artist"].isin(new_playlist_df["artist"])]
filtered_saved_2 = saved_tracks_2[~saved_tracks_2["artist"].isin(new_playlist_df["artist"])]
sample_n = (25-len(new_playlist_df))//2

new_playlist_df = pd.concat([new_playlist_df,filtered_saved_1.sample(sample_n),filtered_saved_2.sample(sample_n)])
new_playlist_df.reset_index(drop = True, inplace= True)

#get seed_tracks from new_playlist_df for recommendations 
seed_tracks = new_playlist_df["track_id"].tolist()
#create recommendations df
recomm_dfs = []
for i in range(5,26,5):
    recomms = sp_1.recommendations(seed_tracks = seed_tracks[i-5:i],limit = 25)
    recomms_df = append_audio_features(create_df_recommendations(recomms),sp_1)
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
print("These are the songs for your new DUO playlist")
print(*zip(new_playlist_df["track_name"],new_playlist_df["artist"]))

#add tracks to playlist
confirm = input("Please confirm that you want to replace the current playlist by typing YES   ")
###CREATE A PLAYLIST BEFORE RUNNING THIS CODE
playlist_id = "-YOUR PLAYLIST ID(URI)-"
if confirm == "YES":
    sp_1.playlist_replace_items(playlist_id=playlist_id,
                              items = new_playlist_df["track_id"].tolist())