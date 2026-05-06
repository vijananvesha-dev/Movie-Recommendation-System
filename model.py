import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# 1. LOAD DATA

DATA_DIR = Path(__file__).resolve().parent / 'archive (3)'
movies = pd.read_csv(DATA_DIR / 'movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(DATA_DIR / 'ratings_small.csv')

# 2. CLEAN DATA

movies = movies[['id', 'title']]
ratings = ratings[['userId', 'movieId', 'rating']]

# Rename columns to match
movies.rename(columns={'id': 'movieId'}, inplace=True)

# Convert IDs to numeric (important)
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')

# Drop invalid rows
movies.dropna(subset=['movieId'], inplace=True)
ratings.dropna(subset=['movieId'], inplace=True)

# Convert to int
movies['movieId'] = movies['movieId'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)

# 3. MERGE DATA

data = pd.merge(ratings, movies, on='movieId')

# 4. CREATE PIVOT TABLE

movie_matrix = data.pivot_table(index='title', columns='userId', values='rating').fillna(0)

# 5. APPLY KNN

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(movie_matrix)

# 6. RECOMMEND FUNCTION
def recommend_movie(movie_name):
    if movie_name not in movie_matrix.index:
        return ["Movie not found"]

    movie_vector = movie_matrix.loc[movie_name].values.reshape(1, -1)

    distances, indices = model.kneighbors(movie_vector, n_neighbors=6)

    recommendations = []
    for i in range(1, len(indices.flatten())):
        recommendations.append(movie_matrix.index[indices.flatten()[i]])

    return recommendations

# 7. TEST (optional)

if __name__ == "__main__":
    print("Sample Movies:")
    print(movie_matrix.index[:10])

    movie = input("Enter movie name: ")
    print("Recommendations:")
    print(recommend_movie(movie))