import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('spotify_data.csv')

print(tracks.head())
tracks.dropna(inplace = True)

tracks = tracks.drop(['artist_name', 'track_name', 'track_id', 'genre'], axis=1)

model = TSNE(n_components = 2, random_state = 0)
tsne_data = model.fit_transform(tracks.head(500))
plt.figure(figsize = (7, 7))
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.show()
