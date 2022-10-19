from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
import pandas as pd
import numpy as np

def getRecommendation(songs_df, tracks_w_features):

    liked_songs_nofeatures = songs_df.loc[songs_df['class'] == 1]
    liked_songs_uris = liked_songs_nofeatures['track_uri'].to_numpy()

    #Now we get uris of the songs the user liked we can get the features from the feature dataframe
    liked_song_features = pd.DataFrame()
    for i in liked_songs_uris:
        song_with_features = tracks_w_features.loc[tracks_w_features['uri'] == i]
        liked_songs_features = liked_song_features.append(song_with_features)

    liked_song_features = liked_songs_features.drop(columns=['uri'])

    average_session_songs = liked_song_features[['acousticness','danceability','energy','instrumentalness','liveness',
                                                      'loudness','speechiness','tempo','valence']].mean()


    return maximal_marginal_relevance(average_session_songs,tracks_w_features.sample(n=20))

def maximal_marginal_relevance(v1,songs, lambda_constant=0.5, threshold_terms=1, sim = True):
    """
    Return ranked phrases using MMR. Cosine similarity is used as similarity measure.
    :param v1: query vector
    :param songs: matrix having index as songs and values as vector
    :param lambda_constant: 0.5 to balance diversity and accuracy. if lambda_constant is high, then higher accuracy. If lambda_constant is low then high diversity.
    :param threshold_terms: number of terms to include in result set
    :return: Ranked songs with score
    """

    s = []
    r = songs['uri'].tolist()
    while len(r) > 0:
        score = 0
        song_to_add = None
        for i in r:
            print(len(r))
            row = songs.loc[songs['uri'] == i]
            row = row.drop(columns=['uri'])
            row = row.to_numpy()
            if len(row) < 1:
              r.remove(i)
              break
            if sim:
                first_part = cosine_similarity([v1], [row[0]])
            else:
                first_part = laplacian_kernel([v1], [row[0]])
            second_part = 0
            for j in s:
                row2 = songs.loc[songs['uri'] == j[0]]
                row2 = row2.drop(columns=['uri'])
                row2 = row2.to_numpy()
                if sim:
                    sim = cosine_similarity([row[0]],[row2[0]])
                else:
                    sim = laplacian_kernel([row[0]], [row2[0]])
                if sim > second_part:
                    second_part = sim
            equation_score = lambda_constant*(first_part)-(1-lambda_constant) * second_part
            if equation_score > score:
                score = equation_score
                song_to_add = i
        if song_to_add is None:
            song_to_add = i
        r.remove(song_to_add)
        s.append((song_to_add, score))
    return (s, s[:threshold_terms])[threshold_terms > len(s)]



