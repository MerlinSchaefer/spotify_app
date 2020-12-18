##imports

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

def authenticate(redirect_uri, client_cred_manager, username, scope,client_id,client_secret):
    """
    Authenticates a user for a given spotify app.

    Parameters
    ----------
    redirect_uri : the redirect uri of the spotify app
    client_cred_manager : SpotifyClientCredentials containing client_id and client_secret
    username: spotify username
    scope: authorization scopes as a string
    client_id: spotify app client_id 
    client_secret: spotify app client_secret

    Returns
    -------
    sp: Spotify auth-token (SpotifyOAuth)

    """

    sp = spotipy.Spotify(client_credentials_manager = client_cred_manager)
    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
    return sp


def create_df_top_songs(api_results):
    """
    Reads in the spotipy query results for user top songs and returns a DataFrame with
    track_name,track_id, artist,album,duration,popularity

    Parameters
    ----------
    api_results : the results of a query to spotify with .current_user_top_tracks()

    Returns
    -------
    df: DataFrame containing track_name,track_id, artist,album,duration,popularity

    """
    #create lists for df-columns
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    #loop through api_results
    for items in api_results['items']:
        try:
            track_name.append(items['name'])
            track_id.append(items['id'])
            artist.append(items["artists"][0]["name"])
            duration.append(items["duration_ms"])
            album.append(items["album"]["name"])
            popularity.append(items["popularity"])
        except TypeError:
            pass
    # Create the final df   
    df = pd.DataFrame({ "track_name": track_name, 
                                "album": album, 
                                "track_id": track_id,
                                "artist": artist, 
                                "duration": duration, 
                                "popularity": popularity})

    return df

def create_df_saved_songs(api_results):
    """
    Reads in the spotipy query results for user saved songs and returns a DataFrame with
    track_name,track_id, artist,album,duration,popularity

    Parameters
    ----------
    api_results : the results of a query to spotify with .current_user_saved_tracks()

    Returns
    -------
    df: DataFrame containing track_name,track_id, artist,album,duration,popularity

    """
    #create lists for df-columns
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    #loop through api_results
    for items in api_results["items"]:
        try:
            track_name.append(items["track"]['name'])
            track_id.append(items["track"]['id'])
            artist.append(items["track"]["artists"][0]["name"])
            duration.append(items["track"]["duration_ms"])
            album.append(items["track"]["album"]["name"])
            popularity.append(items["track"]["popularity"])
        except TypeError: 
            pass
    # Create the final df   
    df = pd.DataFrame({ "track_name": track_name, 
                             "album": album, 
                             "track_id": track_id,
                             "artist": artist, 
                             "duration": duration, 
                             "popularity": popularity})
    return df




def top_artists_from_API(api_results):
    """
    Reads in the spotipy query results for user top artists and returns a DataFrame with
    name, id, genres, popularity and uri

    Parameters
    ----------
    api_results : the results of a query to spotify with .current_user_top_artists()

    Returns
    -------
    df: DataFrame containing name, id, genres, popularity and uri

    """
    df = pd.DataFrame(api_results["items"])
    cols = ["name","id","genres","popularity","uri"]
    return df[cols]
    
def create_df_recommendations(api_results):
    """
    Reads in the spotipy query results for spotify recommended songs and returns a 
    DataFrame with track_name,track_id,artist,album,duration,popularity

    Parameters
    ----------
    api_results : the results of a query to spotify with .recommendations()

    Returns
    -------
    df: DataFrame containing track_name, track_id, artist, album, duration, popularity

    """
    track_name = []
    track_id = []
    artist = []
    album = []
    duration = []
    popularity = []
    for items in api_results['tracks']:
        try:
            track_name.append(items['name'])
            track_id.append(items['id'])
            artist.append(items["artists"][0]["name"])
            duration.append(items["duration_ms"])
            album.append(items["album"]["name"])
            popularity.append(items["popularity"])
        except TypeError:
            pass
        df = pd.DataFrame({ "track_name": track_name, 
                                "album": album, 
                                "track_id": track_id,
                                "artist": artist, 
                                "duration": duration, 
                                "popularity": popularity})

    return df

def create_df_playlist(api_results,sp = None, append_audio = True):
    """
    Reads in the spotipy query results for a playlist and returns a 
    DataFrame with track_name,track_id,artist,album,duration,popularity
    and audio_features unless specified otherwise.

    Parameters
    ----------
    api_results : the results of a query to spotify with .recommendations()
    sp : spotfiy authentication token (result of authenticate())
    append_audio : argument to choose whether to append audio features

    Returns
    -------
    df: DataFrame containing track_name, track_id, artist, album, duration, popularity

    """
    df = create_df_saved_songs(api_results["tracks"])
    if append_audio == True:
        assert sp != None, "sp needs to be specified for appending audio features"
        df = append_audio_features(df,sp)
    return df

def append_audio_features(df,spotify_auth, return_feat_df = False):
    """ 
    Fetches the audio features for all songs in a DataFrame and
    appends these as rows to the DataFrame.
    Requires spotipy to be set up with an auth token.

    Parameters
    ----------
    df : Dataframe containing at least track_name and track_id for spotify songs
    spotify_auth: spotfiy authentication token (result of authenticate())
    return_feat_df: argument to choose whether to also return df with just the audio features
    
    Returns
    -------
    df: DataFrame containing all original rows and audio features for each song

    df_features(optional): DataFrame containing just the audio features
    """
    audio_features = spotify_auth.audio_features(df["track_id"][:])
    assert len(audio_features) == len(df["track_id"][:])
    feature_cols = list(audio_features[0].keys())[:-7]
    features_list = []
    for features in audio_features:
        try:
            song_features = [features[col] for col in feature_cols]
            features_list.append(song_features)
        except TypeError:
            pass
    df_features = pd.DataFrame(features_list,columns = feature_cols)
    df = pd.concat([df,df_features],axis = 1)
    if return_feat_df == False:
        return df
    else:
        return df,df_features

def dataframe_difference(df1, df2, which=None):
    """ 
    Finds rows which are different between two DataFrames.

    Parameters
    ----------
    df1 : Dataframe
    df2 : Dataframe
    which : which rows to keep

    Returns
    -------
    diff_df: DataFrame containing the differences in rows

    """
    comparison_df = df1.merge(
        df2,
        indicator=True,
        how='outer'
    )
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    diff_df.drop("_merge",axis = 1, inplace = True)
    return diff_df.drop_duplicates().reset_index(drop = True)

def create_similarity_score(df1,df2,similarity_score = "cosine_sim"):
    """ 
    Creates a similarity matrix for the audio features of two Dataframes.

    Parameters
    ----------
    df1 : DataFrame containing track_name,track_id, artist,album,duration,popularity
            and all audio features
    df2 : DataFrame containing track_name,track_id, artist,album,duration,popularity
            and all audio features
    
    similarity_score: similarity measure (linear,cosine_sim)

    Returns
    -------
    df: DataFrame containing all original rows and audio features for each song

    df_features(optional): DataFrame containing just the audio features
    """
    
    assert list(df1.columns[6:]) == list(df2.columns[6:])
    features = list(df1.columns[6:])
    df_features1,df_features2 = df1[features],df2[features]
    scaler = StandardScaler()
    df_features_scaled1,df_features_scaled2 = scaler.fit_transform(df_features1),scaler.fit_transform(df_features2)
    #indices = pd.Series(df.index, index = df['track_name']).drop_duplicates()
    if similarity_score == "linear":
        linear_sim = linear_kernel(df_features_scaled1, df_features_scaled2)
        return linear_sim
    elif similarity_score == "cosine_sim":
        cosine_sim = cosine_similarity(df_features_scaled1, df_features_scaled2)
        return cosine_sim
    #other measures may be implemented in the future

def filter_with_meansong(mean_song,recommendations_df, n_recommendations = 10):
    """ 
    Creates a dataframe of final recommendations filtered with a "mean" song from a playlist.

    Parameters
    ----------
    mean_song : DataFrame containing mean values of the original playlist dataframe to represent "mean" song
    recommendations_df : DataFrame containing songs as recommended by spotify including their audio features
    n_recommendations : number of final recommended songs
    
    similarity_score: similarity measure (linear,cosine_sim)

    Returns
    -------
    df: DataFrame containing all original rows and audio features for each song

    df_features(optional): DataFrame containing just the audio features
    """

    mean_song_feat = mean_song[list(mean_song.columns[6:])].values
    mean_song_scaled = StandardScaler().fit_transform(mean_song_feat.reshape(-1,1))
    recommendations_df_scaled = StandardScaler().fit_transform(recommendations_df[list(mean_song.columns[6:])])
    mean_song_scaled = mean_song_scaled.reshape(1,-1)
    sim_mean_finrecomms = cosine_similarity(mean_song_scaled,recommendations_df_scaled)[0][:]
    #sim_mean_finrecomms = sim_mean_finrecomms[0][:]
    indices = (-sim_mean_finrecomms).argsort()[:n_recommendations]
    final_recommendations = recommendations_df.iloc[indices]
    return final_recommendations

###still a work in progress
def get_recommendations(df,song_title, similarity_score, num_recommends = 5):
    """ 
    Gives top num_recommends recommendations for a song based on a similarity_score or matrix

    Parameters
    ----------
    df : DataFrame containing at least track_name

    song_title : Song from df to get recommendations for.

    num_recommends: number of songs to recommend.
    
    similarity_score: similarity matrix

    Returns
    -------
    Song recommendations for song_title
    """
    indices = pd.Series(df.index, index = df['track_name']).drop_duplicates()
    idx = indices[song_title]
    sim_scores = list(enumerate(similarity_score[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1],reverse = True)
    top_scores = sim_scores[1:num_recommends+1]
    song_indices = [i[0] for i in top_scores]
    return df["track_name"].iloc[song_indices]
###still a work in progress