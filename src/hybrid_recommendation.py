import pandas as pd

class HybridRecommendation:
    def __init__(self, content_based_model, collaborative_model, w_cb=0.5, w_cf=0.5):
        self.content_based_model = content_based_model
        self.collaborative_model = collaborative_model
        self.w_cb = w_cb  # Waga modelu content-based
        self.w_cf = w_cf  # Waga modelu collaborative

    def get_hybrid_recommendations(self, user_id, top_n=10):
        """Generuje rekomendacje hybrydowe (średnia ważona) za pomocą outer join."""

        # 1. Rekomendacje z modelu Content-Based Filtering
        print("Generowanie rekomendacji Content-Based Filtering...")
        user_movie_data = self.content_based_model.prepare_user_movie_data()  # przygotowanie danych
        user_movies = user_movie_data[user_movie_data['userId'] == user_id]
        features = user_movies.drop(columns=['userId', 'movieId', 'rating', 'liked', 'title', 'timestamp'])

        # Przewidywanie, czy użytkownik polubi dany film
        content_based_scores = self.content_based_model.model.predict_proba(features)[:, 1]
        user_movies = user_movies.copy()
        user_movies.loc[:, 'content_based_score'] = content_based_scores

        # 2. Rekomendacje z modelu Collaborative Filtering
        print("Generowanie rekomendacji Collaborative Filtering...")
        recommendations_collaborative = self.collaborative_model.get_user_recommendations(user_id, top_n)

        # 3. Łączenie rekomendacji z obu modeli za pomocą outer join
        content_based_recommendations = user_movies[['movieId', 'content_based_score']].copy()
        recommendations_collaborative = recommendations_collaborative[['movieId', 'similarity']]

        recommendations = pd.merge(
            content_based_recommendations,
            recommendations_collaborative,
            on='movieId',
            how='outer'
        )

        # Zamiast używania inplace=True, używamy przypisania do kolumn
        recommendations['content_based_score'] = recommendations['content_based_score'].fillna(0)
        recommendations['similarity'] = recommendations['similarity'].fillna(0)

        # Obliczanie średniej ważonej (według wag modelu)
        recommendations['hybrid_score'] = (
            self.w_cb * recommendations['content_based_score'] +
            self.w_cf * recommendations['similarity']
        )

        # Sortowanie rekomendacji według hybrid_score
        recommendations = recommendations.sort_values(by='hybrid_score', ascending=False)

        return recommendations[['movieId', 'hybrid_score']].head(top_n)
