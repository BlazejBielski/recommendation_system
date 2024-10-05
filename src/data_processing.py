import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import fetch_and_unpack_files_from_url


movielens_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
processed_data_dir = 'data/processed'


def load_data():
    data_dir = fetch_and_unpack_files_from_url(movielens_url, source_name="ml-latest-small")

    if os.path.exists(os.path.join(data_dir, 'ml-latest-small')):
        data_dir = os.path.join(data_dir, 'ml-latest-small')

    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    tags = pd.read_csv(os.path.join(data_dir, 'tags.csv'))
    links = pd.read_csv(os.path.join(data_dir, 'links.csv'))

    print(f"Load ratings: {ratings.shape}, movies: {movies.shape}, tags: {tags.shape}, links: {links.shape}")

    return ratings, movies, tags, links


def clean_data(data):
    print("Data cleaning...")
    cleaned_data = data.dropna().drop_duplicates()
    print(f"After data cleaning: {cleaned_data.shape}")
    return cleaned_data


def normalize_data(data, columns):
    print(f"Normalize columns: {columns}")
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


def split_data(data, test_size=0.2):
    print(f"Split data on test and training (test_size={test_size})...")
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    print(f"Training data: {train.shape}, Test data: {test.shape}")
    return train, test


def save_processed_data(data, filename):
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    file_path = os.path.join(processed_data_dir, filename)
    data.to_csv(file_path, index=False)
    print(f"Zapisano przetworzone dane do {file_path}")


if __name__ == '__main__':

    ratings, movies, tags, links = load_data()

    cleaned_ratings = clean_data(ratings)
    cleaned_movies = clean_data(movies)
    cleaned_tags = clean_data(tags)
    cleaned_links = clean_data(links)

    normalized_ratings = normalize_data(cleaned_ratings, ['rating'])

    train_data, test_data = split_data(normalized_ratings)

    save_processed_data(train_data, 'ratings_train.csv')
    save_processed_data(test_data, 'ratings_test.csv')
    save_processed_data(cleaned_movies, 'movies_clean.csv')
    save_processed_data(cleaned_tags, 'tags_clean.csv')
    save_processed_data(cleaned_links, 'links_clean.csv')
