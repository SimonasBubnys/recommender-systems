from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def recommend(average_session_songs, playlist):
    """
            :param average_session_songs: Songs that the user has listened until now
            :param playlist: Songs to rank
            """

    result = maximal_marginal_relevance(average_session_songs,playlist)
    return result

def maximal_marginal_relevance(v1,songs, lambda_constant=0.5, threshold_terms=10):
    """
    Return ranked phrases using MMR. Cosine similarity is used as similarity measure.
    :param v1: query vector
    :param songs: matrix having index as songs and values as vector
    :param lambda_constant: 0.5 to balance diversity and accuracy. if lambda_constant is high, then higher accuracy. If lambda_constant is low then high diversity.
    :param threshold_terms: number of terms to include in result set
    :return: Ranked songs with score
    """
    s = []
    r = list(songs.iloc[:,0])
    while len(r) > 0:
        score = 0
        song_to_add = None
        for i in r:
            row = songs.loc[songs['id'] == i]
            row = row.to_numpy()
            first_part = cosine_similarity([v1], [row[0][1:]])
            second_part = 0
            for j in s:
                row2 = songs.loc[songs['id'] == j[0]]
                row2 = row2.to_numpy()
                cos_sim = cosine_similarity([row[0][1:]],[row2[0][1:]])
                if cos_sim > second_part:
                    second_part = cos_sim
            equation_score = lambda_constant*(first_part)-(1-lambda_constant) * second_part
            if equation_score > score:
                score = equation_score
                song_to_add = i
        if song_to_add is None:
            song_to_add = i
        r.remove(song_to_add)
        s.append((song_to_add, score))
    return (s, s[:threshold_terms])[threshold_terms > len(s)]

"""
EXAMPLE 
In this implementation it is assumed that the songs will be in a pandas dataframe and the first colummn is the index 
of the songs, but it can be changed later
"""
if __name__ == "__main__":
    average_session_songs  = np.array([0.63,0.616,1,-8.128,0,0.0309,0.463,0.0408,0.173,0.509,118.65])
    s1 =  np.array([0,0.23,0.84,1,-8.128,0,0.034,0.99,0.01,0.173,0.509,118.65])
    s2 = np.array([1,0.23,0.06,1,-8.128,0,0.109,0.99,0.02,0.2,0.7,111.65])
    s3 = np.array([2,0.25,0.52,1,-8.128,0,0.043,0.45,0.03,0.3,0.9,200.65])
    s4 = np.array([3,0.89,0.31,1,-8.128,0,0.041,0.22,0.04,0.4,0.1,50.65])
    playlist = pd.DataFrame([s1,s2,s3,s4])
    playlist.columns = ['id', '', '', '','','','','','','','','']
    recommend = recommend(average_session_songs,playlist)
    print(recommend)


