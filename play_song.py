import spotipy
from spotipy.oauth2 import SpotifyClientCredentials as cc


if __name__ == "__main__":
    client_id = '898dd71dc932407e85921f0ac79f0127'
    client_secret = '16d994f070064371beb8758f32d64180'
    # https://developer.spotify.com/dashboard/applications/898dd71dc932407e85921f0ac79f0127
    client_cc = cc(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_cc)


    urn = 'spotify:artist:4FXGRMSHh2JjHxVwS8dhH1'

    artist = sp.artist(urn)
    #print(artist['name'])

    uri = 'spotify:track:0na4kLr2ixGkEA7N16IkkV'
    track = sp.track(uri)
    print(track['name'])

