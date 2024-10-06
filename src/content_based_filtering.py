import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from data_processing import load_data

processed_data_dir = 'data/processed'


def preprocess_movie_features(movies):
    """Process the features of the videos, e.g., extract genres and convert to vectors."""
    print("Processing of film features...")

    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies.index)
    movies = pd.concat([movies, genre_df], axis=1)

    return movies.drop(columns=['genres'])


def prepare_user_movie_data(ratings, movies):
    """Combine user ratings with video features to create a dataset for training."""
    print("Combining user ratings with video features...")
    user_movie_data = pd.merge(ratings, movies, on='movieId')
    user_movie_data['liked'] = user_movie_data['rating'].apply(lambda x: 1 if x >= 3.5 else 0)

    return user_movie_data


def train_content_based_model(user_movie_data):
    """Train the Random Forest model to predict whether a user will like a particular video."""
    print("Training the classification model...")

    features = user_movie_data.drop(columns=['userId', 'movieId', 'rating', 'liked', 'title', 'timestamp'])
    target = user_movie_data['liked']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    return model


def save_processed_data(data, filename):
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    file_path = os.path.join(processed_data_dir, filename)
    data.to_csv(file_path, index=False)
    print(f"Saved the processed data to {file_path}")


if __name__ == '__main__':
    ratings, movies, tags, links = load_data()
    processed_movies = preprocess_movie_features(movies)
    user_movie_data = prepare_user_movie_data(ratings, processed_movies)
    save_processed_data(user_movie_data, 'user_movie_data.csv')
    trained_model = train_content_based_model(user_movie_data)
