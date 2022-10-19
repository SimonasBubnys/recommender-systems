from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import laplacian_kernel
import pandas as pd
import numpy as np

def getRecommendation(songs):
    liked_songs = songs.loc[songs['class'] == 1]
    average_session_songs = liked_songs[['acousticness','danceability','energy','liveness','tempo','valence']].mean()
    songs_to_recommend_from = songs.sample(n=20)
    songs_to_recommend_from =songs_to_recommend_from.filter(['acousticness','danceability','energy','liveness','tempo','valence','track_uri'])
    return maximal_marginal_relevance(average_session_songs,songs_to_recommend_from)

def maximal_marginal_relevance(v1,songs_to_compare, lambda_constant=0.5, threshold_terms=1, sim = True):
    """
    Return ranked phrases using MMR. Cosine similarity is used as similarity measure.
    :param v1: query vector
    :param songs: matrix having index as songs and values as vector
    :param lambda_constant: 0.5 to balance diversity and accuracy. if lambda_constant is high, then higher accuracy. If lambda_constant is low then high diversity.
    :param threshold_terms: number of terms to include in result set
    :return: Ranked songs with score
    """
    s = []
    r = songs_to_compare['track_uri'].tolist()
    while len(r) > 0:
        score = 0
        song_to_add = None
        for i in r:
            row = songs_to_compare.loc[songs_to_compare['track_uri'] == i]
            row = row.drop(columns=['track_uri'])
            row = row.to_numpy()
            print(row)
            if len(row) < 1:
              r.remove(i)
              break
            if sim:
                first_part = cosine_similarity([v1], [row[0]])
            else:
                first_part = laplacian_kernel([v1], [row[0]])
            second_part = 0
            for j in s:
                row2 = songs_to_compare.loc[songs_to_compare['track_uri'] == j[0]]
                row2 = row2.drop(columns=['track_uri'])
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
    return s[:threshold_terms][0][0]




