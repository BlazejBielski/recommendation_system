import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFiltering:
    def __init__(self, ratings, movies):
        self.ratings = ratings
        self.movies = movies
        self.user_similarity = None
        self.item_similarity = None
        self.user_recommendations = None

    def compute_user_similarity(self):
        """Calculates the similarity matrix of users."""
        user_ratings = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_similarity = cosine_similarity(user_ratings)
        return self.user_similarity

    def compute_item_similarity(self):
        """Calculates the similarity matrix of objects (movies)."""
        item_ratings = self.ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        self.item_similarity = cosine_similarity(item_ratings)
        return self.item_similarity

    def get_user_recommendations(self, user_id, top_n=5):
        """Generates recommendations for a given user."""
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        similar_users = pd.DataFrame(self.user_similarity[user_id - 1], columns=['similarity'])
        similar_users['userId'] = self.ratings['userId'].unique()
        similar_users = similar_users.sort_values(by='similarity', ascending=False).head(top_n)

        # Link similar users with their ratings
        recommendations = pd.merge(similar_users, self.ratings, on='userId', how='left')
        recommendations = recommendations[recommendations['rating'].notnull()]

        # Filter already rated videos
        rated_movie_ids = user_ratings['movieId'].unique()
        recommendations = recommendations[~recommendations['movieId'].isin(rated_movie_ids)]

        # Sort recommendations by similarity
        return recommendations.sort_values(by='similarity', ascending=False)[['movieId', 'similarity']].head(top_n)


if __name__ == '__main__':
    # Loading the processed data
    ratings = pd.read_csv('data/processed/ratings_train.csv')  # lub inny plik, który pasuje
    movies = pd.read_csv('data/processed/movies_clean.csv')

    # Initialization of the recommendation system
    cf = CollaborativeFiltering(ratings, movies)

    # Calculating similarity
    cf.compute_user_similarity()

    # An example of a recommendation for a user with ID 1
    recommendations = cf.get_user_recommendations(user_id=1)
    print("Recommendation for user with ID 1:")
    print(recommendations)
