import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.load_data()
        
    def load_data(self):
        movies = pd.read_csv('data/tmdb_5000_movies.csv')
        credits = pd.read_csv('data/tmdb_5000_credits.csv')
        movies = movies.merge(credits, on='title')
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        movies = movies.dropna()
        
        movies['tags'] = movies['overview'] + ' ' + movies['genres'] + ' ' + \
                         movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['crew']
        
        self.movies = movies[['movie_id', 'title', 'tags']]
        self.movies.loc[:, 'tags'] = self.movies['tags'].str.lower()
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['tags'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
    
    def get_recommendations(self, title, top_n=10):
        try:
            idx = self.movies[self.movies['title'] == title].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:top_n+1]]
            return self.movies.iloc[top_indices]['title'].tolist()
        except:
            return []

    def suggest_similar_titles(self, query, n=5):
        return get_close_matches(query, self.movies['title'].tolist(), n=n, cutoff=0.6)

    def fallback_recommendations(self):
        return ['Inception', 'The Matrix', 'Titanic', 'Parasite', 'The Godfather']

    def get_movie_titles(self):
        return self.movies['title'].tolist()

recommender = MovieRecommender()