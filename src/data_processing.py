import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from utils import fetch_and_unpack_files_from_url

movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
processed_data_dir = 'data/processed'

def load_data():
    """Ładuje dane z zestawu MovieLens, zawsze ściągając i przetwarzając je na nowo."""
    data_dir = fetch_and_unpack_files_from_url(movielens_url, source_name="ml-latest-small")
    if os.path.exists(os.path.join(data_dir, 'ml-latest-small')):
        data_dir = os.path.join(data_dir, 'ml-latest-small')

    # Ładowanie plików CSV po przetworzeniu
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    tags = pd.read_csv(os.path.join(data_dir, 'tags.csv'))
    links = pd.read_csv(os.path.join(data_dir, 'links.csv'))

    print(f"Załadowano dane: ratings {ratings.shape}, movies {movies.shape}, tags {tags.shape}, links {links.shape}")
    return ratings, movies, tags, links


def clean_data(data):
    """Czyści dane poprzez usunięcie duplikatów i wartości NaN."""
    cleaned_data = data.dropna().drop_duplicates()
    print(f"Po oczyszczeniu: {cleaned_data.shape}")
    return cleaned_data


def normalize_data(data, columns):
    """Normalizuje wskazane kolumny w zestawie danych."""
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def split_data(data, test_size=0.2):
    """Dzieli dane na zestawy treningowy i testowy."""
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    print(f"Zestaw treningowy: {train.shape}, Zestaw testowy: {test.shape}")
    return train, test

def save_processed_data(data, filename):
    """Zapisuje przetworzone dane do pliku CSV w katalogu `processed_data_dir`."""
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    file_path = os.path.join(processed_data_dir, filename)
    data.to_csv(file_path, index=False)
    print(f"Zapisano przetworzone dane do {file_path}")


def ensure_data_availability():
    """Sprawdza, czy dane zostały już przetworzone, jeśli nie, przetwarza je."""
    print("Zawsze ściągamy i przetwarzamy dane od nowa...")

    # Ładujemy dane z oryginalnego źródła
    ratings, movies, tags, links = load_data()

    # Oczyszczamy dane
    cleaned_ratings = clean_data(ratings)
    cleaned_movies = clean_data(movies)

    # Normalizujemy oceny
    normalized_ratings = normalize_data(cleaned_ratings, ['rating'])

    # Dzielimy dane na zestawy treningowy i testowy
    train_data, test_data = split_data(normalized_ratings)

    # Zapisujemy przetworzone dane
    save_processed_data(train_data, 'ratings_train.csv')
    save_processed_data(test_data, 'ratings_test.csv')
    save_processed_data(cleaned_movies, 'movies_clean.csv')
