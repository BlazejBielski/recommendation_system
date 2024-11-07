import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_processing import ensure_data_availability

class CollaborativeFiltering:
    def __init__(self):
        ensure_data_availability()  # Zapewnienie dostępności danych
        self.ratings = pd.read_csv('data/processed/ratings_train.csv')
        self.movies = pd.read_csv('data/processed/movies_clean.csv')
        self.user_similarity = None
        self.item_similarity = None

    def compute_user_similarity(self):
        """Oblicza macierz podobieństwa użytkowników."""
        user_ratings = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_similarity = cosine_similarity(user_ratings)
        return self.user_similarity

    def compute_item_similarity(self):
        """Oblicza macierz podobieństwa filmów."""
        item_ratings = self.ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.item_similarity = cosine_similarity(item_ratings)
        return self.item_similarity

    def get_user_recommendations(self, user_id, top_n=5):
        """Generuje rekomendacje dla danego użytkownika."""
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        similar_users = pd.DataFrame(self.user_similarity[user_id - 1], columns=['similarity'])
        similar_users['userId'] = self.ratings['userId'].unique()
        similar_users = similar_users.sort_values(by='similarity', ascending=False).head(top_n)

        # Połączenie podobnych użytkowników z ich ocenami
        recommendations = pd.merge(similar_users, self.ratings, on='userId', how='left')
        recommendations = recommendations[recommendations['rating'].notnull()]

        # Filtracja już ocenionych filmów
        rated_movie_ids = user_ratings['movieId'].unique()
        recommendations = recommendations[~recommendations['movieId'].isin(rated_movie_ids)]

        # Posortowanie rekomendacji według podobieństwa
        recommendations = recommendations.sort_values(by='similarity', ascending=False)
        return recommendations[['movieId', 'similarity']].head(top_n)


