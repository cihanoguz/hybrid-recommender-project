import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Set up paths
# Get the directory where this script is located (data/)
base_dir = os.path.dirname(os.path.abspath(__file__))

# CSV files are in the same directory as this script (data/)
movie_path = os.path.join(base_dir, "movie.csv")
rating_path = os.path.join(base_dir, "rating.csv")

print("📂 movie_path:", movie_path)
print("📂 rating_path:", rating_path)

# 2. Read data
movie = pd.read_csv(movie_path)
rating = pd.read_csv(
    rating_path, dtype={"userId": "int32", "movieId": "int32", "rating": "float32"}
)

print("👉 movie shape:", movie.shape)
print("👉 rating shape:", rating.shape)

# 3. Merge
df = movie.merge(rating, how="left", on="movieId")
print("👉 merged df shape:", df.shape)

# 4. Filter rare movies (movies with less than 1000 ratings)
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
print("👉 common_movies shape:", common_movies.shape)

# 5. User-movie pivot table
user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
print("👉 user_movie_df shape:", user_movie_df.shape)

# 6. 🎯 GENRE-BASED COSINE SIMILARITY (BEFORE PICKLE!)
print("\n🔄 Calculating genre-based similarity...")

# Clean NaN genres
movie["genres"] = movie["genres"].fillna("Unknown")

# Process genres ("|" → space)
movie["genres_processed"] = movie["genres"].str.replace("|", " ", regex=False)

# Create binary matrix using CountVectorizer
cv_genre = CountVectorizer(token_pattern=r"\b\w+\b", lowercase=True)
genre_matrix = cv_genre.fit_transform(movie["genres_processed"])

# Calculate Cosine Similarity
cosine_sim_genre = cosine_similarity(genre_matrix, genre_matrix)

print(f"✅ cosine_sim_genre shape: {cosine_sim_genre.shape}")

# 7. Save as Parquet (optional - save to data directory)
rating.to_parquet(os.path.join(base_dir, "rating.parquet"))
user_movie_df.to_parquet(os.path.join(base_dir, "user_movie_df.parquet"))
common_movies.to_parquet(os.path.join(base_dir, "common_movies.parquet"))

print("✅ Parquet files saved successfully:")
print("   - rating.parquet")
print("   - user_movie_df.parquet")
print("   - common_movies.parquet")

# 8. 🎯 SAVE TO PICKLE (INCLUDING cosine_sim_genre!)
data_dict = {
    "movie": movie,
    "rating": rating,
    "df_full": df,
    "common_movies": common_movies,
    "user_movie_df": user_movie_df,
    "cosine_sim_genre": cosine_sim_genre,  # ← NEWLY ADDED
}

# Save pickle file in the same directory as this script
pickle_path = os.path.join(base_dir, "prepare_data.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(data_dict, f)

print("✅ Prepared data file saved as 'prepare_data.pkl'.")

# 9. 🔍 VERIFICATION
print("\n" + "=" * 50)
print("🔍 VERIFICATION STARTING...")
print("=" * 50)

with open(pickle_path, "rb") as f:
    test_data = pickle.load(f)

print("\n📦 Keys in pickle:")
for key in test_data.keys():
    if key == "cosine_sim_genre":
        print(f"  ✅ {key} → shape: {test_data[key].shape}")
    else:
        print(f"  ✓ {key}")

if "cosine_sim_genre" in test_data:
    print(f"\n🎉 SUCCESS! cosine_sim_genre saved to pickle!")
else:
    print("\n❌ ERROR: cosine_sim_genre could not be saved to pickle!")

print(f"\n💾 Pickle file location:")
print(f"   {pickle_path}")
print("=" * 50)
