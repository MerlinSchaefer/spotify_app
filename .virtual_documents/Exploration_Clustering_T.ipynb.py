import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

username = usernames[1]



scope = "user-library-read user-read-recently-played user-top-read playlist-modify-public playlist-read-private playlist-read-collaborative"

redirect_uri = "https://developer.spotify.com/dashboard/applications/4a4e029d299a4241873db8300038bf0a"

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                      client_secret=client_secret)


sp = authenticate(redirect_uri, client_credentials_manager, username, scope, client_id, client_secret)


top_tracks_short = sp.current_user_top_tracks(limit = 50,offset=0,time_range='short_term')
top_tracks_med = sp.current_user_top_tracks(limit = 50,offset=0,time_range='medium_term')
top_tracks_long = sp.current_user_top_tracks(limit = 50,offset=0,time_range='long_term')

#combine the top_tracks
top_tracks_short_df = append_audio_features(create_df_top_songs(top_tracks_short),sp)
top_tracks_med_df = append_audio_features(create_df_top_songs(top_tracks_med),sp)
top_tracks_long_df = append_audio_features(create_df_top_songs(top_tracks_long),sp)

top_tracks_short_df["Timeframe"] = "short term"
top_tracks_med_df["Timeframe"] = "medium term"
top_tracks_long_df["Timeframe"] = "long term"

top_tracks = pd.concat([top_tracks_short_df, top_tracks_med_df, top_tracks_long_df])
top_tracks = top_tracks.reset_index(drop = True)

top_artists_long = sp.current_user_top_artists(limit = 50, time_range = "long_term")
top_artists_med = sp.current_user_top_artists(limit = 50, time_range = "medium_term")
top_artists_short = sp.current_user_top_artists(limit = 50, time_range = "short_term")


artists_short_df = top_artists_from_API(top_artists_short)
artists_med_df = top_artists_from_API(top_artists_med)
artists_long_df = top_artists_from_API(top_artists_long)


artists_df = pd.concat([artists_short_df,artists_med_df,artists_long_df])
artists_df["genres"] = artists_df["genres"].apply(lambda x: ",".join(x))
artists_df.drop_duplicates().reset_index(drop = True)


top_tracks.drop_duplicates()


list(top_tracks_short_df.columns[6:-2])


audio_features = list(top_tracks_short_df.columns[6:-2])
audio_features.append("Timeframe")


audio_overtime_df = top_tracks[audio_features].groupby("Timeframe").mean()
audio_overtime_df


audio_overtime_df.reset_index(inplace = True)
df_loud_key = audio_overtime_df[["loudness","key"]] #for better viz these need to be plotted seperately
df_loud_key.set_index(pd.Index(["long term", "medium term", "short term"]), inplace = True)


audio_overtime_df.drop(["loudness","key"], axis = 1, inplace = True)


plot_df = pd.melt(audio_overtime_df, id_vars = "Timeframe",
                           var_name = "audio_feature", value_name = "mean")
plot_df


#get_ipython().run_line_magic("matplotlib", " inline")
sns.catplot(data = plot_df, kind = "bar", x = "Timeframe",
            y = "mean", hue = "audio_feature"
)
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.show()


cluster_feat = audio_features[:-1]
cluster_feat


df_cluster = top_tracks[cluster_feat].drop("mode",axis = 1)
df_cluster.hist(figsize= (15,10))


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = np.array(df_cluster)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


ss_dist = []
for k in range(1,21):
    km = KMeans(n_clusters=k,max_iter = 10000 ,random_state=13)
    km = km.fit(X)
    ss_dist.append(km.inertia_)
    
plt.plot(range(1,21), ss_dist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()




from sklearn.metrics import silhouette_score
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
plt.figure(figsize=(8, 3))
sns.lineplot(x = range(2, 10), y = silhouette_scores,marker="o")


n_cluster = 6
km = KMeans(n_clusters = n_cluster, max_iter = 10000, random_state=13).fit(X)


X_pred = km.predict(X)


X_pred.shape


top_tracks["cluster"] = X_pred


for i in range(1,n_cluster+1):
    print("\n",top_tracks[["track_name","artist"]][top_tracks["cluster"] == i].drop_duplicates(), "\n")



from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=50, n_iter=5000, random_state=13)
tsne_results = tsne.fit_transform(X)

df_tsne = pd.DataFrame(tsne_results)
df_tsne.columns = ['D1', 'D2']
df_tsne['label'] = km.labels_
df_tsne.head()


sns.scatterplot(data = df_tsne, x = "D1", y = "D2", hue = "label", style = "label",
                palette = "tab10")
plt.legend()
plt.show()


from sklearn.decomposition import PCA
pca = PCA(n_components=3, random_state=123)
pca_results = pca.fit_transform(X)
print(pca.explained_variance_ratio_.sum())
pca.explained_variance_ratio_.cumsum()


tsne3 = TSNE(n_components=3, n_iter=5000, random_state=13, perplexity=50)
tsne_results = tsne3.fit_transform(X)


df_tsne3 = pd.DataFrame(tsne_results)
df_tsne3.columns = ['D1', 'D2', 'D3']
df_tsne3['label'] = km.labels_
df_tsne3.head()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(df_tsne3['D1'], df_tsne3['D2'], df_tsne3['D3'],
           c=df_tsne3['label'], cmap='tab10')

ax.set_xlabel('D1')
ax.set_ylabel('D2')
ax.set_zlabel('D3')
plt.show()


from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=2, kernel = "rbf", gamma = 0.04)
rbf_pca_results = rbf_pca.fit_transform(X)

df_rbf_pca = pd.DataFrame(rbf_pca_results)
df_rbf_pca.columns = ['D1', 'D2']
df_rbf_pca['label'] = km.labels_
df_rbf_pca.head()


sns.scatterplot(data = df_rbf_pca, x = "D1", y = "D2", hue = "label", style = "label",
                palette = "tab10")
plt.legend()
plt.show()


from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=5)
lle_results = lle.fit_transform(X)

df_lle = pd.DataFrame(lle_results)
df_lle.columns = ['D1', 'D2']
df_lle['label'] = km.labels_
df_lle.head()


sns.scatterplot(data = df_lle, x = "D1", y = "D2", hue = "label", style = "label",
                palette = "tab10")
plt.legend()
plt.show()


#average song feature per cluster (what are our clusters representing?)
df_cluster


df_cluster["cluster"] = X_pred
df_cluster.drop(["key","loudness"],axis = 1,inplace = True)


df_radar = df_cluster.groupby("cluster").mean().reset_index()
df_radar


# https://python-graph-gallery.com/392-use-faceting-for-radar-chart/
from math import pi, ceil

def make_radar(row, title, color, dframe, num_clusters):
    # number of variable
    categories=list(dframe)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the radar plot
    ax = plt.subplot(2,ceil(num_clusters/2),row+1, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=8)
    plt.ylim(0,1)

    # Ind1
    values=dframe.loc[row].drop('cluster').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=16, color=color, y=1.06)


plt.figure(figsize = (20,10))
my_palette = plt.cm.get_cmap("tab10", len(df_radar.index))
title_list = ["acoustic", "instrumental dance","dance","dance again","live dance","acoustic chill"]

for row in range(0, len(df_radar.index)):
    make_radar(row=row, title= title_list[row], 
               color=my_palette(row), dframe=df_radar, num_clusters=len(df_radar.index))






