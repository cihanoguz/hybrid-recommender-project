import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Yol kur
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(
    os.path.join(base_dir, "..", "datasets", "movie_lens_dataset")
)

movie_path = os.path.join(data_dir, "data/movie.csv")
rating_path = os.path.join(data_dir, "data/rating.csv")

print("📂 movie_path:", movie_path)
print("📂 rating_path:", rating_path)

# 2. Veri oku
movie = pd.read_csv(movie_path)
rating = pd.read_csv(
    rating_path,
    dtype={
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32"
    }
)

print("👉 movie shape:", movie.shape)
print("👉 rating shape:", rating.shape)

# 3. Birleştir
df = movie.merge(rating, how="left", on="movieId")
print("👉 merged df shape:", df.shape)

# 4. Nadir filmleri ele (1000 altı oy alan filmler)
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
print("👉 common_movies shape:", common_movies.shape)

# 5. Kullanıcı-film pivotu
user_movie_df = common_movies.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)
print("👉 user_movie_df shape:", user_movie_df.shape)

# 6. 🎯 GENRE-BASED COSINE SIMILARITY (PICKLE'DAN ÖNCE!)
print("\n🔄 Genre-based similarity hesaplanıyor...")

# NaN genre'leri temizle
movie['genres'] = movie['genres'].fillna('Unknown')

# Genre'leri işle ("|" → boşluk)
movie['genres_processed'] = movie['genres'].str.replace('|', ' ', regex=False)

# CountVectorizer ile binary matrix oluştur
cv_genre = CountVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
genre_matrix = cv_genre.fit_transform(movie['genres_processed'])

# Cosine Similarity hesapla
cosine_sim_genre = cosine_similarity(genre_matrix, genre_matrix)

print(f"✅ cosine_sim_genre shape: {cosine_sim_genre.shape}")

# 7. Parquet olarak kaydet
rating.to_parquet(os.path.join(data_dir, "rating.parquet"))
user_movie_df.to_parquet(os.path.join(data_dir, "user_movie_df.parquet"))
common_movies.to_parquet(os.path.join(data_dir, "common_movies.parquet"))

print("✅ Parquet dosyaları başarıyla kaydedildi:")
print("   - rating.parquet")
print("   - user_movie_df.parquet")
print("   - common_movies.parquet")

# 8. 🎯 PICKLE'A KAYDET (cosine_sim_genre DAHİL!)
data_dict = {
    "movie": movie,
    "rating": rating,
    "df_full": df,
    "common_movies": common_movies,
    "user_movie_df": user_movie_df,
    "cosine_sim_genre": cosine_sim_genre  # ← YENİ EKLENEN
}

with open("prepare_data.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("✅ Hazır veri dosyası 'prepare_data.pkl' olarak kaydedildi.")

# 9. 🔍 DOĞRULAMA
print("\n" + "="*50)
print("🔍 DOĞRULAMA BAŞLIYOR...")
print("="*50)

with open("prepare_data.pkl", "rb") as f:
    test_data = pickle.load(f)

print("\n📦 Pickle içindeki key'ler:")
for key in test_data.keys():
    if key == "cosine_sim_genre":
        print(f"  ✅ {key} → shape: {test_data[key].shape}")
    else:
        print(f"  ✓ {key}")

if 'cosine_sim_genre' in test_data:
    print(f"\n🎉 BAŞARILI! cosine_sim_genre pickle'a kaydedildi!")
else:
    print("\n❌ HATA: cosine_sim_genre pickle'a kaydedilemedi!")

print(f"\n💾 Pickle dosyası konumu:")
print(f"   {os.path.abspath('prepare_data.pkl')}")
print("="*50)