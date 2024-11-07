from src.collaborative_filtering import CollaborativeFiltering
from src.content_based_filtering import ContentBasedFiltering
from src.hybrid_recommendation import HybridRecommendation
from data_processing import ensure_data_availability, load_data

def main():
    # Ładowanie danych (ratings i movies)
    ratings, movies, _, _ = load_data()

    # 1. Tworzenie instancji Content-Based Filtering
    print("Uruchamianie modelu Content-Based Filtering...")
    content_based_model = ContentBasedFiltering(ratings, movies)

    # Przetwarzanie cech filmów
    processed_movies = content_based_model.preprocess_movie_features()
    # Łączenie ocen użytkowników z cechami filmów
    user_movie_data = content_based_model.prepare_user_movie_data()
    # Trenowanie modelu
    trained_model = content_based_model.train_model(user_movie_data)

    # Generowanie rekomendacji dla przykładowego użytkownika (np. ID=1) w modelu Content-Based
    content_based_recommendations = content_based_model.get_user_recommendations(user_id=1, top_n=10)
    print(f"\nRekomendacje dla użytkownika o ID 1 (Content-Based Filtering):")
    print(content_based_recommendations[['movieId', 'content_based_score']])

    # 2. Tworzenie instancji Collaborative Filtering
    print("\nUruchamianie modelu Collaborative Filtering...")
    collaborative_model = CollaborativeFiltering()
    collaborative_model.compute_user_similarity()
    collaborative_model.compute_item_similarity()

    # Generowanie rekomendacji dla przykładowego użytkownika (np. ID=1) w modelu Collaborative Filtering
    recommendations_collaborative = collaborative_model.get_user_recommendations(user_id=1, top_n=10)
    print(f"\nRekomendacje dla użytkownika o ID 1 (Collaborative Filtering):")
    print(recommendations_collaborative[['movieId', 'similarity']])

    # 3. Tworzenie modelu Hybrydowego
    print("\nUruchamianie modelu Hybrydowego...")
    hybrid_model = HybridRecommendation(content_based_model, collaborative_model, w_cb=0.6, w_cf=0.4)

    # Generowanie rekomendacji hybrydowych dla użytkownika o ID 1
    hybrid_recommendations = hybrid_model.get_hybrid_recommendations(user_id=1, top_n=10)
    print(f"\nRekomendacje dla użytkownika o ID 1 (Hybrid Model):")
    print(hybrid_recommendations)

if __name__ == '__main__':
    ensure_data_availability()  # Zapewnienie dostępności danych przed uruchomieniem modelu
    main()
