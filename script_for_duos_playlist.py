##script for duos playlist

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd
from spotifuncs import append_audio_features, create_df_from_API, create_similarity_score, get_recommendations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
