import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from data_processing import load_data

processed_data_dir = 'data/processed'


class ContentBasedFiltering:
    def __init__(self, ratings, movies):
        """Initialize the content-based filtering model with user ratings and movie data."""
        self.ratings = ratings
        self.movies = movies
        self.model = None

    def preprocess_movie_features(self):
        """Process the features of the movies, e.g., extracts genres and converts them to vectors."""
        print("Processing movie features...")

        # Split genres into separate columns
        self.movies['genres'] = self.movies['genres'].apply(lambda x: x.split('|'))

        # Convert genres to binary form (MultiLabelBinarizer)
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.movies['genres'])

        # Create a new DataFrame with binary genre columns
        genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=self.movies.index)

        # Combine genres with the rest of the movie data
        self.movies = pd.concat([self.movies, genre_df], axis=1)

        # Remove the 'genres' column as it is no longer needed
        self.movies = self.movies.drop(columns=['genres'])

        # Now make sure the conversion to numeric values does not include text columns (e.g., 'title')
        # List of columns that should not be converted to numbers
        non_numeric_cols = ['title', 'movieId']

        # Ensure that all other columns are numeric
        numeric_cols = self.movies.columns.difference(non_numeric_cols)
        self.movies[numeric_cols] = self.movies[numeric_cols].apply(pd.to_numeric, errors='coerce')

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

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

        return self.model

    def save_processed_data(self, data, filename):
        """Save processed data to a CSV file."""
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)

        file_path = os.path.join(processed_data_dir, filename)
        data.to_csv(file_path, index=False)
        print(f"Saved the processed data to {file_path}")


if __name__ == '__main__':
    ratings, movies, tags, links = load_data()
    cbf = ContentBasedFiltering(ratings, movies)

    # Process movie features
    processed_movies = cbf.preprocess_movie_features()

    # Combine user ratings with movie features
    user_movie_data = cbf.prepare_user_movie_data()

    # Save processed data
    cbf.save_processed_data(user_movie_data, 'user_movie_data.csv')

    # Train the model
    trained_model = cbf.train_model(user_movie_data)
