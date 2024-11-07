import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.data_processing import processed_data_dir


class ContentBasedFiltering:
    def __init__(self, ratings, movies):
        """Initialize the content-based filtering model with user ratings and movie data."""
        self.ratings = ratings
        self.movies = movies
        self.model = None

    def preprocess_movie_features(self):
        print("Processing movie features...")

        # Użyj gatunków
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.movies['genres'])
        genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=self.movies.index)

        # Dodaj cechy związane z popularnością
        self.movies['mean_rating'] = self.movies['movieId'].map(self.ratings.groupby('movieId')['rating'].mean())

        # Łączenie wszystkich cech w jeden DataFrame
        self.movies = pd.concat([self.movies, genre_df], axis=1)
        self.movies = self.movies.drop(columns=['genres'])

        return self.movies

    def prepare_user_movie_data(self):
        """Combine user ratings with movie features to create a dataset for training."""
        print("Combining user ratings with movie features...")
        user_movie_data = pd.merge(self.ratings, self.movies, on='movieId')
        user_movie_data['liked'] = user_movie_data['rating'].apply(lambda x: 1 if x >= 3.5 else 0)

        return user_movie_data

    def train_model(self, user_movie_data):
        """Train the Random Forest model to predict whether a user will like a particular movie."""
        print("Training the classification model...")

        features = user_movie_data.drop(columns=['userId', 'movieId', 'rating', 'liked', 'title', 'timestamp'])
        target = user_movie_data['liked']

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

        return self.model

    def get_user_recommendations(self, user_id, top_n=10):
        """Generate recommendations for a given user using the trained content-based model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Filtrujemy filmy, których użytkownik jeszcze nie ocenił
        rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].values
        unrated_movies = self.movies[~self.movies['movieId'].isin(rated_movies)]

        # Przygotowujemy cechy dla nieocenionych filmów
        features = unrated_movies.drop(columns=['title', 'movieId'])

        # Przewidujemy prawdopodobieństwo, że użytkownik polubi każdy z nieocenionych filmów
        predicted_scores = self.model.predict_proba(features)[:, 1]  # Druga kolumna to prawdopodobieństwo klasy "1" (lubi)

        # Tworzymy DataFrame z wynikami
        recommendations = unrated_movies[['movieId']].copy()
        recommendations['content_based_score'] = predicted_scores

        # Sortujemy rekomendacje według przewidywanego prawdopodobieństwa i zwracamy top N
        recommendations = recommendations.sort_values(by='content_based_score', ascending=False)

        return recommendations.head(top_n)
