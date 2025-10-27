import pickle
import pandas as pd
import numpy as np
import os

# -------------------------------------------------
# AYARLAR
# -------------------------------------------------
PICKLE_IN = "prepare_data.pkl"          # büyük orijinal pickle
PICKLE_OUT = "prepare_data_demo.pkl"    # küçük/demo pickle
USER_ID_DEMO = 108170                   # göstereceğim kullanıcı ID'si
MIN_OVERLAP_RATIO = 0.60                # user-based için overlap eşiği
TOP_REF_FILM_COUNT = 30                 # item-based için en benzer film sayısı
TOP_CONTENT_FILM_COUNT = 100            # content-based için top benzer film sayısı


# -------------------------------------------------
# 1. ORİJİNAL VERİYİ YÜKLE
# -------------------------------------------------
with open(PICKLE_IN, "rb") as f:
    data = pickle.load(f)

movie_full = data["movie"].copy()
rating_full = data["rating"].copy()
df_full_full = data["df_full"].copy()
common_movies_full = data["common_movies"].copy()
user_movie_df_full = data["user_movie_df"].copy()
cosine_sim_genre_full = data["cosine_sim_genre"].copy()

# -------------------------------------------------
# 2. DEMO KULLANICININ DÜNYASINI ÇIKAR
# -------------------------------------------------
if USER_ID_DEMO not in user_movie_df_full.index:
    raise ValueError(f"Demo kullanıcısı {USER_ID_DEMO} user_movie_df'te yok.")

row = user_movie_df_full.loc[[USER_ID_DEMO]]
watched_movies_titles = row.columns[row.notna().any()].to_list()
num_watched = len(watched_movies_titles)
print("Demo kullanıcı kaç film izlemiş:", num_watched)

demo_user_ratings = rating_full[rating_full["userId"] == USER_ID_DEMO].copy()

# -------------------------------------------------
# 3. USER-BASED: BENZER KULLANICILAR
# -------------------------------------------------
watched_df = user_movie_df_full[watched_movies_titles]
user_movie_count = watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userId", "movie_count"]

threshold_common = num_watched * MIN_OVERLAP_RATIO
candidate_users = user_movie_count[
    (user_movie_count["userId"] != USER_ID_DEMO) &
    (user_movie_count["movie_count"] > threshold_common)
]["userId"].unique().tolist()

keep_users_user_based = set(candidate_users + [USER_ID_DEMO])
print("Aday kullanıcı sayısı:", len(candidate_users))

# -------------------------------------------------
# 4. ITEM-BASED: REFERANS FİLM VE BENZERLERİ
# -------------------------------------------------
demo_5stars = demo_user_ratings[demo_user_ratings["rating"] == 5.0].copy()
if not demo_5stars.empty:
    if "timestamp" in demo_5stars.columns:
        last_fav_row = demo_5stars.sort_values("timestamp", ascending=False).iloc[0]
    else:
        last_fav_row = demo_5stars.iloc[0]
    ref_movie_id = last_fav_row["movieId"]
    ref_movie_title_arr = movie_full.loc[movie_full["movieId"] == ref_movie_id, "title"].values
    if len(ref_movie_title_arr) > 0:
        ref_movie_title = ref_movie_title_arr[0]
        if ref_movie_title in user_movie_df_full.columns:
            ref_vector = user_movie_df_full[ref_movie_title]
            sims = user_movie_df_full.corrwith(ref_vector).dropna()
            sims = sims[sims.index != ref_movie_title].sort_values(ascending=False)
            top_similar_titles = sims.head(TOP_REF_FILM_COUNT).index.tolist()
        else:
            ref_movie_title = None
            top_similar_titles = []
    else:
        ref_movie_title = None
        top_similar_titles = []
else:
    ref_movie_title = None
    top_similar_titles = []

print("Referans film:", ref_movie_title)
print("Item-based top benzer film sayısı:", len(top_similar_titles))

# -------------------------------------------------
# 5. CONTENT-BASED: TÜRLERE GÖRE BENZER FİLMLER
# -------------------------------------------------
keep_titles_initial = set(watched_movies_titles) | set(top_similar_titles)
title_to_idx = {t: i for i, t in enumerate(movie_full['title'].tolist())}
expanded_titles = set(keep_titles_initial)

for t in list(keep_titles_initial):
    if t in title_to_idx:
        idx = title_to_idx[t]
        sim_scores = list(enumerate(cosine_sim_genre_full[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_neighbors_idx = [i for i, _ in sim_scores[:TOP_CONTENT_FILM_COUNT]]
        for ni in top_neighbors_idx:
            expanded_titles.add(movie_full.iloc[ni]['title'])

print("Film sayısı (expanded_titles):", len(expanded_titles))

# -------------------------------------------------
# 6. FİLTRELEME
# -------------------------------------------------
user_movie_df_small = user_movie_df_full.loc[
    user_movie_df_full.index.isin(keep_users_user_based),
    user_movie_df_full.columns.isin(expanded_titles)
].copy()

keep_movies_df = movie_full[movie_full["title"].isin(expanded_titles)].copy()
keep_movie_ids = keep_movies_df["movieId"].unique().tolist()

rating_small = rating_full[
    (rating_full["userId"].isin(keep_users_user_based)) &
    (rating_full["movieId"].isin(keep_movie_ids))
].copy()

df_full_small = df_full_full[
    (df_full_full["userId"].isin(keep_users_user_based)) &
    (df_full_full["movieId"].isin(keep_movie_ids))
].copy()

common_movies_small = common_movies_full[
    (common_movies_full["userId"].isin(keep_users_user_based)) &
    (common_movies_full["movieId"].isin(keep_movie_ids))
].copy()

# -------------------------------------------------
# 7. COSINE SIMILARITY MATRİSİNİ KÜÇÜLT
# -------------------------------------------------
keep_idx = [title_to_idx[t] for t in movie_full['title'].tolist() if t in expanded_titles]
cosine_sim_genre_small = cosine_sim_genre_full[np.ix_(keep_idx, keep_idx)]
movie_small = movie_full.iloc[keep_idx].reset_index(drop=True)

missing_titles = [
    c for c in user_movie_df_small.columns
    if c not in movie_small["title"].values
]
if missing_titles:
    print("UYARI: user_movie_df_small içinde movie_small'ta olmayan title var:", missing_titles)

# -------------------------------------------------
# 8. YENİ PICKLE KAYDET
# -------------------------------------------------
data_small = {
    "movie": movie_small,
    "rating": rating_small,
    "df_full": df_full_small,
    "common_movies": common_movies_small,
    "user_movie_df": user_movie_df_small,
    "cosine_sim_genre": cosine_sim_genre_small
}

with open(PICKLE_OUT, "wb") as f:
    pickle.dump(data_small, f)

print("✅ DONE. Küçültülmüş pickle kaydedildi:", PICKLE_OUT)

"""
USER_ID_DEMO = 108170                   # göstereceğim kullanıcı ID'si
MIN_OVERLAP_RATIO = 0.60                # user-based için overlap eşiği
TOP_REF_FILM_COUNT = 30                 # item-based için en benzer film sayısı
TOP_CONTENT_FILM_COUNT = 100            # content-based için top benzer film sayısı

Demo kullanıcı kaç film izlemiş: 186
Aday kullanıcı sayısı: 2325
Referans film: Wild at Heart (1990)
Item-based top benzer film sayısı: 30
Film sayısı (expanded_titles): 7395

"""

"""
USER_ID_DEMO = 108170                   # göstereceğim kullanıcı ID'si
MIN_OVERLAP_RATIO = 0.20                # user-based için overlap eşiği
TOP_REF_FILM_COUNT = 50                 # item-based için en benzer film sayısı
TOP_CONTENT_FILM_COUNT = 200            # content-based için top benzer film sayısı

Demo kullanıcı kaç film izlemiş: 186
Aday kullanıcı sayısı: 30291
Referans film: Wild at Heart (1990)
Item-based top benzer film sayısı: 50
Film sayısı (expanded_titles): 11042
"""