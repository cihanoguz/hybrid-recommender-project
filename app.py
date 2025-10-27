# -------------------------------------------------
# HYBRID RECOMMENDER STREAMLIT APP
# Eğitim Sunumu için Optimize Edilmiş Versiyon
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------------------------------
# SAYFA AYARLARI (GENEL GÖRÜNÜM)
# -------------------------------------------------
st.set_page_config(
    page_title="Hybrid Recommender Case Study",
    layout="wide"
)

# Basit ama okunabilir custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 12px 24px -12px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
        min-height: 5.5rem;
    }
    .metric-title {
        font-weight: 500;
        font-size: .8rem;
        color: #6b7280;
        margin-bottom: .25rem;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.4rem;
        word-break: break-word;
    }

    .header-badge-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem 1rem;
        margin-bottom: .75rem;
    }
    .header-badge {
        background: #fdf8c7;
        color: #111827;
        display: inline-block;
        padding: .4rem .6rem;
        border-radius: .25rem;
        font-size: .8rem;
        font-weight: 500;
        border: 1px solid #e2e0a8;
    }

    table.var-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        background: #ffffff;
        color: #111827;
    }
    table.var-table th {
        text-align: left;
        background: #fdf8c7;
        color: #111827;
        font-weight: 600;
        padding: .5rem .6rem;
        border: 1px solid #d4d4d4;
        white-space: nowrap;
    }
    table.var-table td {
        padding: .5rem .6rem;
        border: 1px solid #d4d4d4;
        vertical-align: top;
        background: #ffffff;
        color: #111827;
    }

    table.stage-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 1rem;
        font-size: .85rem;
        background: #ffffff;
        color: #111827;
    }
    table.stage-table th {
        text-align: left;
        background: #eef2ff;
        color: #111827;
        font-weight: 600;
        padding: .5rem .6rem;
        border: 1px solid #c7c9df;
        white-space: nowrap;
    }
    table.stage-table td {
        padding: .5rem .6rem;
        border: 1px solid #c7c9df;
        vertical-align: top;
        background: #ffffff;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# VERİYİ YÜKLEME
# -------------------------------------------------
# Burada varsayıyoruz ki senin daha önce hazırladığın pickle dosyası
# şu anahtarlarla döndürüyordu:
# movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre
#
# NOT: Bu pickle'ı küçültüp (subset) sınıf/demolar için optimize etmen
# performansı ciddi iyileştirir. Bu, sonuçları mantıksal olarak bozmaz;
# sadece işlem hacmini makul boyuta indirir.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICKLE_PATH = os.path.join(BASE_DIR, "data/prepare_data_demo.pkl")


@st.cache_resource(show_spinner=True)
def load_data(pickle_path: str):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    movie = data["movie"]
    rating = data["rating"]
    df_full = data["df_full"]
    common_movies = data["common_movies"]
    user_movie_df = data["user_movie_df"]
    cosine_sim_genre = data["cosine_sim_genre"]

    # Kullanıcı ID listesini matristen alıyoruz
    all_user_ids = user_movie_df.index.tolist()

    return movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre

movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre = load_data(PICKLE_PATH)


# -------------------------------------------------
# PERFORMANS İÇİN ÖN-HESAPLAMA YAKLAŞIMI
# -------------------------------------------------
# Buradaki fikir şu:
# - Ağır işi (benzer kullanıcıları bulma, korelasyonları hesaplama vb.)
#   kullanıcı bazında tek defa yap.
# - Slider parametreleri ile sadece filtrele/sırala gibi hafif işlemler yap.
#
# Bu, canlıda slider oynarken uygulamanın akıcı kalmasını sağlar.




@st.cache_data(show_spinner=False)
def precompute_for_user_userbased(chosen_user: int):
    """
    User-Based Recommendation için ağır hazırlık.

    - movies_watched: hedef kullanıcının izlediği filmlerin listesi
    - candidate_users_df: her başka kullanıcının bu filmlerden kaçını izlediği
    - corr_df: hedef kullanıcı ile diğer kullanıcıların korelasyonları
    - top_users_ratings: bu kullanıcıların (komşu adayların) hangi filmlere kaç puan verdiği

    Bu fonksiyonun amacı finalize aşamasına düzgün veri sağlamak.
    """

    # Kullanıcı gerçekten var mı?
    if chosen_user not in user_movie_df.index:
        return {
            "status": "no_user",
            "movies_watched": [],
            "candidate_users_df": pd.DataFrame(),
            "corr_df": pd.DataFrame(),
            "top_users_ratings": pd.DataFrame(),
        }

    # Hedef kullanıcının izlediği filmleri bul
    row = user_movie_df.loc[[chosen_user]]
    movies_watched = row.columns[row.notna().any()].to_list()

    if len(movies_watched) == 0:
        return {
            "status": "no_movies",
            "movies_watched": [],
            "candidate_users_df": pd.DataFrame(),
            "corr_df": pd.DataFrame(),
            "top_users_ratings": pd.DataFrame(),
        }

    # Hedef kullanıcının izlediği filmler üzerinden alt matris:
    # satır = userId, sütun = bu filmler
    movies_watched_df = user_movie_df[movies_watched].copy()

    # 1) ORTAK İZLEME SAYISI (CRITICAL FIX)
    # Her kullanıcı bu filmlerden kaçını izlemiş?
    # Bu kez satır bazında sayıyoruz: axis=1
    user_movie_count_series = movies_watched_df.notnull().sum(axis=1)

    candidate_users_df = (
        user_movie_count_series
        .reset_index()  # index -> userId geliyor
        .rename(columns={"index": "userId", 0: "movie_count"})
    )
    candidate_users_df.columns = ["userId", "movie_count"]

    # kendimizi çıkar
    candidate_users_df = candidate_users_df[candidate_users_df["userId"] != chosen_user].copy()

    # 2) KORELASYON
    # base_vector: hedef kullanıcının rating vektörü (tüm filmler üzerinden)
    base_vector = user_movie_df.loc[chosen_user]

    # movies_watched_df.T: film -> user
    # corrwith(base_vector) user bazlı korelasyon döndürüyor (index=userId)
    corr_series = movies_watched_df.T.corrwith(base_vector).dropna()

    corr_df = (
        corr_series
        .reset_index()
        .rename(columns={"index": "userId", 0: "corr"})
    )
    corr_df.columns = ["userId", "corr"]

    # kendimizi çıkaralım
    corr_df = corr_df[corr_df["userId"] != chosen_user].copy()

    if corr_df.empty:
        return {
            "status": "no_corr",
            "movies_watched": movies_watched,
            "candidate_users_df": candidate_users_df,
            "corr_df": corr_df,
            "top_users_ratings": pd.DataFrame(),
        }

    # 3) KOMŞU ADAYLARIN RATINGLERİ
    # komşu adayların puan verdiği bütün filmId'leri ve puanlarını çekiyoruz
    # burada corr_df ile merge yaparak corr bilgisini kaybetmiyoruz
    # (böylece 'corr' kolonunu sonradan da kullanabileceğiz)
    top_users_ratings = corr_df.merge(
        rating[["userId", "movieId", "rating"]],
        on="userId",
        how="inner"
    )

    if top_users_ratings.empty:
        return {
            "status": "no_ratings_from_neighbors",
            "movies_watched": movies_watched,
            "candidate_users_df": candidate_users_df,
            "corr_df": corr_df,
            "top_users_ratings": top_users_ratings,
        }

    return {
        "status": "ok",
        "movies_watched": movies_watched,
        "candidate_users_df": candidate_users_df,  # userId, movie_count
        "corr_df": corr_df,                        # userId, corr
        "top_users_ratings": top_users_ratings,    # userId, movieId, rating, corr
    }



def finalize_user_based_from_cache(
    precomputed,
    min_overlap_ratio_pct: float,
    corr_threshold: float,
    max_neighbors: int,
    weighted_score_threshold: float,
    top_n: int,
    chosen_user: int
):
    """
    precompute çıktısını alır ve:
    1. overlap filtresi
    2. corr filtresi
    3. max_neighbors
    4. corr * rating ile ağırlıklandırma
    5. kullanıcı zaten izlemişse atma
    6. weighted_score_threshold ile süzme
    """

    status = precomputed["status"]
    if status != "ok":
        return {
            "status": status,
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": precomputed.get("candidate_users_df", pd.DataFrame()),
            "dbg_corr_df": precomputed.get("corr_df", pd.DataFrame()),
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    movies_watched = precomputed["movies_watched"]
    candidate_users_df = precomputed["candidate_users_df"].copy()
    corr_df = precomputed["corr_df"].copy()
    top_users_ratings = precomputed["top_users_ratings"].copy()

    # Savunma
    if len(movies_watched) == 0 or candidate_users_df.empty or corr_df.empty or top_users_ratings.empty:
        return {
            "status": "not_enough_data",
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 1. overlap filtresi
    # Örn: min_overlap_ratio_pct = 20 ise → "benim izlediklerimin en az %20'sini izlemiş ol"
    threshold_common = len(movies_watched) * (min_overlap_ratio_pct / 100.0)

    good_overlap_users = candidate_users_df[
        candidate_users_df["movie_count"] >= threshold_common
    ]["userId"]

    if good_overlap_users.empty:
        return {
            "status": "no_candidates_after_overlap",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": 0,
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 2. korelasyon filtresi
    corr_filtered = corr_df[
        (corr_df["userId"].isin(good_overlap_users)) &
        (corr_df["corr"] >= corr_threshold)
    ].copy()

    if corr_filtered.empty:
        return {
            "status": "no_similar_users",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 3. max_neighbors uygula
    corr_filtered = (
        corr_filtered
        .sort_values("corr", ascending=False)
        .head(max_neighbors)
        .copy()
    )

    if corr_filtered.empty:
        return {
            "status": "no_similar_users_after_limit",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 4. seçilen komşuların film ratinglerini al ve corr ile birleştir
    neighbor_ratings = top_users_ratings.merge(
        corr_filtered[["userId", "corr"]],
        on="userId",
        how="inner"
    )

    if neighbor_ratings.empty:
        return {
            "status": "no_neighbor_ratings_after_filter",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": len(corr_filtered["userId"].unique()),
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": neighbor_ratings,
        }

    # 5. ağırlıklı puan = rating * corr (kolon adı her zaman aynı olmayabiliyor)
    possible_corr_cols = [c for c in neighbor_ratings.columns if "corr" in c]
    if not possible_corr_cols:
        return {
            "status": "no_corr_column_after_merge",
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": neighbor_ratings,
        }

    corr_col = possible_corr_cols[0]
    neighbor_ratings["weighted_rating"] = (
            neighbor_ratings["rating"] * neighbor_ratings[corr_col]
    )

    recommendation_df = (
        neighbor_ratings
        .groupby("movieId")
        .agg(weighted_rating=("weighted_rating", "mean"))
        .reset_index()
    )

    # Hedef kullanıcının zaten izlediklerini çıkar
    seen_ids = rating.loc[rating["userId"] == chosen_user, "movieId"].unique().tolist()
    recommendation_df = recommendation_df[
        ~recommendation_df["movieId"].isin(seen_ids)
    ]

    # weighted score threshold uygula
    recommendation_df = recommendation_df[
        recommendation_df["weighted_rating"] >= weighted_score_threshold
    ]

    # sırala + top_n
    recommendation_df = (
        recommendation_df
        .sort_values("weighted_rating", ascending=False)
        .head(top_n)
        .copy()
    )

    # film isimleri
    recommendation_df = recommendation_df.merge(
        movie[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    out_df = recommendation_df[["title", "weighted_rating"]].rename(
        columns={"title": "Film", "weighted_rating": "Skor"}
    )

    debug_info = {
        "movies_watched": len(movies_watched),
        "candidate_users": len(candidate_users_df),
        "after_overlap_users": len(good_overlap_users),
        "after_corr_users": len(corr_filtered["userId"].unique()),
        "used_neighbors": len(corr_filtered["userId"].unique())
    }

    return {
        "status": "ok",
        "recommendations": out_df,
        "debug_info": debug_info,
        "dbg_candidate_users_df": candidate_users_df,
        "dbg_corr_df": corr_df,
        "dbg_corr_filtered": corr_filtered,
        "dbg_neighbor_ratings": neighbor_ratings,
    }


@st.cache_data(show_spinner=False)
def precompute_for_user_itembased(chosen_user: int):
    """
    Item-Based Recommendation için ağır kısım.
    Kullanıcının en son 5★ verdiği filmi bul
    ve o filme benzer filmlerin korelasyon skorlarını hesapla.
    """

    # Bu kullanıcının 5 verdiği filmler
    user_5 = rating[
        (rating["userId"] == chosen_user) &
        (rating["rating"] == 5.0)
    ]

    if user_5.empty:
        return {
            "status": "no_five_star",
            "reference_movie": None,
            "similarity_df": pd.DataFrame()
        }

    # en son verdiği 5★
    if "timestamp" in user_5.columns:
        last_fav = user_5.sort_values("timestamp", ascending=False).iloc[0]
    else:
        last_fav = user_5.iloc[0]

    ref_movie_id = last_fav["movieId"]
    ref_title_arr = movie.loc[movie["movieId"] == ref_movie_id, "title"].values

    if len(ref_title_arr) == 0:
        return {
            "status": "no_title",
            "reference_movie": None,
            "similarity_df": pd.DataFrame()
        }

    ref_title = ref_title_arr[0]

    # user_movie_df'in kolonları film adları olduğu varsayımıyla ilerliyoruz.
    if ref_title not in user_movie_df.columns:
        return {
            "status": "not_in_matrix",
            "reference_movie": ref_title,
            "similarity_df": pd.DataFrame()
        }

    ref_vector = user_movie_df[ref_title]
    sims = user_movie_df.corrwith(ref_vector).dropna()  # film-film benzerliği

    # kendisini çıkar
    sims = sims[sims.index != ref_title]

    similarity_df = (
        sims.sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "Benzer Film", 0: "Benzerlik"})
    )

    return {
        "status": "ok",
        "reference_movie": ref_title,
        "similarity_df": similarity_df
    }


def finalize_item_based_from_cache(precomputed_item, top_n_item: int):
    """
    Item-based sonuçlarını hafifçe kesip döner.
    """
    status = precomputed_item["status"]
    if status != "ok":
        return status, None, pd.DataFrame()

    ref_movie = precomputed_item["reference_movie"]
    sim_df_all = precomputed_item["similarity_df"]

    sim_df_head = sim_df_all.head(top_n_item).copy()

    return "ok", ref_movie, sim_df_head


@st.cache_data(show_spinner=False)
def content_based_recommender_cached(movie_title: str, top_n: int):
    """
    Tür benzerliğine göre içerik tabanlı öneri (Bonus Görev 3).
    cosine_sim_genre matrisini kullanıyoruz.
    """
    # Film var mı?
    if movie_title not in movie['title'].values:
        return {
            "status": "not_found",
            "reference_movie": movie_title,
            "reference_genres": None,
            "recommendations": pd.DataFrame()
        }

    # Film indeksini bul
    movie_idx = movie[movie['title'] == movie_title].index[0]

    # Referans filmin türlerini al
    ref_genres = movie.iloc[movie_idx]['genres']

    # Tüm filmlerle cosine similarity skorlarını al
    sim_scores = list(enumerate(cosine_sim_genre[movie_idx]))

    # Skora göre sırala, kendisini at
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    # İlgili film indeksleri
    movie_indices = [i[0] for i in sim_scores]

    # Sonuç dataframe
    result_df = movie.iloc[movie_indices][['title', 'genres']].copy()
    result_df['Benzerlik Skoru'] = [round(i[1], 3) for i in sim_scores]
    result_df = result_df.rename(columns={'title': 'Film', 'genres': 'Türler'})

    return {
        "status": "ok",
        "reference_movie": movie_title,
        "reference_genres": ref_genres,
        "recommendations": result_df
    }


# -------------------------------------------------
# YARDIMCI FONKSİYONLAR (METRİKLER İÇİN)
# -------------------------------------------------
def get_matrix_shape(df):
    return df.shape  # (n_users, n_movies)


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_problem, tab_dataset, tab_tasks = st.tabs([
    "1. İş Problemi",
    "2. Veri Seti Hikayesi",
    "3. Proje Görevleri"
])

# -------------------------------------------------
# TAB 1: İŞ PROBLEMİ
# -------------------------------------------------
with tab_problem:
    st.title("Case Study: Hybrid Recommender Project")
    st.header("İş Problemi")

    st.info(
        "💡 Gerçek Dünya Senaryosu: 'ID'si verilen kullanıcı için user-based ve item-based tavsiye yöntemlerini kullanarak film öner.' "
        "Bu problem MovieLens verisi üzerinden tanımlandı. :contentReference[oaicite:0]{index=0}"
    )

    st.write(
        """
        Amaç:
        1. Kullanıcıya benzeyen kullanıcıların sevdiği filmleri öner (User-Based).
        2. Kullanıcının en son 5⭐ verdiği filme benzeyen filmleri öner (Item-Based).
        3. BONUS: Seçilen bir filmin türsel benzerliğine göre benzer filmleri bul (Content-Based).
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧑‍🤝‍🧑 User-Based (Benim Gibi Kullanıcılar)
        - Benimle benzer zevkte kullanıcıları bul
        - Onların sevdiği ama benim izlemediğim filmleri getir
        - Ağırlıklı skora göre sırala
        """)

    with col2:
        st.markdown("""
        ### 🎬 Item-Based (Bu Filme Benzeyenler)
        - En son 5⭐ verdiğim filmi bul
        - Bu filme benzeyen filmleri korelasyonla hesapla
        - En benzerleri sırala
        """)

    with col3:
        st.markdown("""
        ### 🏷️ Content-Based (Benzer Türde Filmler)
        - Bir film seç
        - Tür bilgisine bak
        - Cosine similarity ile aynı tatta filmleri getir
        """)

    st.success(
        "🎯 Hibrit Bakış: Gerçek hayatta bu üç yaklaşım birlikte kullanılarak güçlü bir öneri motoru kurulur."
    )

# -------------------------------------------------
# TAB 2: VERİ SETİ HİKAYESİ
# -------------------------------------------------
with tab_dataset:
    st.title("Veri Seti Hikayesi")

    total_users_full = df_full["userId"].nunique()
    total_movies_full = df_full["movieId"].nunique()
    total_ratings_full = df_full.shape[0]

    total_users_common = common_movies["userId"].nunique()
    total_movies_common = common_movies["movieId"].nunique()
    total_ratings_common = common_movies.shape[0]

    um_users, um_movies = get_matrix_shape(user_movie_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Toplam Kullanıcı (ham)</div>"
            f"<div class='metric-value'>{total_users_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Toplam Film (ham)</div>"
            f"<div class='metric-value'>{total_movies_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Toplam Rating (ham)</div>"
            f"<div class='metric-value'>{total_ratings_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.write(
        "Bu veri seti MovieLens tarafından sağlandı. Yaklaşık on binlerce film ve milyonlarca değerlendirme içeriyor. "
        "Her kullanıcı en az 20 filme oy vermiş durumda; zaman aralığı 1995-2015. :contentReference[oaicite:1]{index=1}"
    )

    st.subheader("Değişkenler")

    st.markdown("**movie.csv**")
    st.markdown(
        """
        <div class='header-badge-wrap'>
            <div class='header-badge'>3 Değişken</div>
            <div class='header-badge'>~27K Gözlem</div>
            <div class='header-badge'>Film bilgileri</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <table class="var-table">
        <tr><th>movieId</th><td>Eşsiz film numarası.</td></tr>
        <tr><th>title</th><td>Film adı.</td></tr>
        <tr><th>genres</th><td>Tür bilgisi (Action|Comedy|Drama ...)</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**rating.csv**")
    st.markdown(
        """
        <div class='header-badge-wrap'>
            <div class='header-badge'>4 Değişken</div>
            <div class='header-badge'>~20M Gözlem</div>
            <div class='header-badge'>Kullanıcı puanları</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <table class="var-table">
        <tr><th>userId</th><td>Kullanıcı ID'si (benzersiz)</td></tr>
        <tr><th>movieId</th><td>Film ID'si</td></tr>
        <tr><th>rating</th><td>Verilen puan</td></tr>
        <tr><th>timestamp</th><td>Verildiği zaman</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Veri Hazırlama Adımları")

    st.markdown(
        """
        MovieLens verisini doğrudan kullanamayız çünkü çok büyük.  
        Bu yüzden üç aşamalı bir yol izliyoruz:

        1. Tüm evren (orijinal, ~20M rating / ~138K user / ~27K film)  
        2. Popüler filmler filtresi (az oy alan filmleri at)  
        3. Eğitim demosu için küçültülmüş snapshot (2.3K user seviyesine kadar daralt)

        Aşağıdaki tablo iki dünyayı yan yana gösteriyor:
        """
    )

    st.markdown(
        f"""
        <table class="stage-table">
        <tr>
            <th>Aşama</th>
            <th>Açıklama</th>
            <th>Rating Satırı</th>
            <th>Film Sayısı</th>
            <th>Kullanıcı Sayısı</th>
        </tr>

        <tr>
            <td>Ham Veri (MovieLens Orijinal)</td>
            <td>movie.merge(rating)<br/>1995-2015 arası oylar<br/>Her kullanıcı ≥20 film oylamış</td>
            <td>{20_000_263:,}+</td>
            <td>~27,000</td>
            <td>~138,000</td>
        </tr>

        <tr>
            <td>Popüler Filmlerle Süzülmüş (Orijinal Mantık)</td>
            <td>1000'in altında oy alan filmleri çıkar<br/>Seyrek / gürültülü filmler elendi</td>
            <td>{17_766_015:,}</td>
            <td>~3,000 civarı aktif film</td>
            <td>~138,000</td>
        </tr>

        <tr>
            <td>Demo Full (Streamlit'te kullandığımız çekirdek)</td>
            <td>Örnek kullanıcı etrafındaki etkileşimleri tutan snapshot</td>
            <td>{1_793_782:,}</td>
            <td>{6_818:,}</td>
            <td>{2_326:,}</td>
        </tr>

        <tr>
            <td>Demo Popüler Filmler (common_movies)</td>
            <td>Yine 1000 altı oy alan filmler atıldı</td>
            <td>{1_572_589:,}</td>
            <td>{1_986:,}</td>
            <td>{2_326:,}</td>
        </tr>

        <tr>
            <td>Demo Kullanıcı-Film Matrisi (user_movie_df)</td>
            <td>pivot: kullanıcı x film (rating matrisi)</td>
            <td>{2_326:,} satır x {1_982:,} sütun</td>
            <td>{1_982:,} aktif film</td>
            <td>{2_326:,} aktif kullanıcı</td>
        </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "💡 Az oylanan filmleri atmak, sistemi hızlandırır ve daha güvenilir benzerlik hesapları yapmamızı sağlar. "
        "Bu da canlı demo sırasında istediğimiz parametrelerle rahat oynamamıza izin veriyor."
    )

# -------------------------------------------------
# TAB 3: PROJE GÖREVLERİ (CANLI DEMO)
# -------------------------------------------------
with tab_tasks:
    st.title("Proje Görevleri ve Canlı Öneri Motoru")

    st.subheader("Görevlerin İş Mantığı")
    st.markdown("""
    Bu case çalışmasında 3 yaklaşım gösteriyoruz:

    **Görev 1: User-Based (Benim Gibi Kullanıcılar)**  
    - Hedef kullanıcının izlediği filmlere benzer zevke sahip diğer kullanıcıları buluyoruz.  
    - Bu benzer kullanıcıların sevdiği ama hedef kullanıcının izlemediği filmleri öneriyoruz.  
    - Korelasyon (corr) puanını 'zevk benzerliği' olarak kullanıyoruz.  
    - Weighted Score = (benzerlik * rating) ortalaması. Bu bize en mantıklı önerileri veriyor. :contentReference[oaicite:3]{index=3}

    **Görev 2: Item-Based (Bu Filme Benzeyenler)**  
    - Hedef kullanıcının en son 5⭐ verdiği filmi buluyoruz.  
    - Bu filmle korelasyon açısından en çok benzerlik gösteren filmleri buluyoruz.  
    - Yani 'Bu filmi sevenler şunları da sevdi' mantığı. :contentReference[oaicite:4]{index=4}

    **Bonus Görev 3: Content-Based (Benzer Türde Filmler)**  
    - Kullanıcı davranışına bakmıyoruz.  
    - Sadece filmlerin içerik özelliklerine (özellikle genres / tür bilgisi) bakıyoruz.  
    - Tür vektörleri arasında cosine similarity (kosinüs benzerliği) hesaplıyoruz.  
    - Sonuç: 'Bu filmin tür DNA'sına en çok benzeyen diğer filmler.'  
    """)

    st.info(
        "Gerçek dünyada hibrit yaklaşım bu üç fikri birleştirir: "
        "topluluk zevki (user-based), ürün benzerliği (item-based), içerik benzerliği (content-based)."
    )

    # ---------------------------------
    # SOL KOLON: PARAMETRELER / KONTROLLER
    # ---------------------------------
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### Parametreler / Kontrol Paneli")

        chosen_user = st.number_input(
            "Hedef Kullanıcı ID",
            min_value=1,
            value=108170,
            step=1,
            help="Case boyunca gösterdiğimiz örnek kullanıcı ID'si."
        )

        rec_type = st.radio(
            "Hangi yöntemi deneyelim?",
            [
                "User-Based (Benim Gibi Kullanıcılar)",
                "Item-Based (Bu Filme Benzeyenler)",
                "Content-Based (Benzer Türde Filmler)"
            ],
            help=(
                "User-Based: 'Benim gibi kullanıcılar ne izliyor?'\n"
                "Item-Based: 'Bu filmi sevdiysen şunları da seversin.'\n"
                "Content-Based: 'Bu filmin tür DNA'sına benzeyen filmler.'"
            ),
            key="rec_type_radio"
        )

        st.markdown("---")

        # Her yaklaşımın kendi parametreleri
        if rec_type.startswith("User-Based"):
            st.markdown("#### 🧑‍🤝‍🧑 User-Based Parametreleri")

            min_overlap_ratio_pct = st.slider(
                "Ortak izleme yüzdesi (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                help="Benim izlediğim filmlerin en az %60'ını izlemiş kullanıcıları 'benzer' kabul et."
            )

            corr_threshold = st.slider(
                "Korelasyon (zevk benzerliği) eşiği",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="0.65 ve üzeri: gerçekten bana benziyor."
            )

            max_neighbors = st.slider(
                "Maksimum komşu sayısı",
                min_value=1,
                max_value=200,
                value=7,
                step=1,
                help="En fazla kaç benzer kullanıcı kullanalım?"
            )

            weighted_score_threshold = st.slider(
                "Weighted skor eşiği",
                min_value=0.0,
                max_value=5.0,
                value=3.5,
                step=0.1,
                help="(corr * rating) ortalaması 3.5 üstüyse öner."
            )

            top_n_user_based = st.slider(
                "Kaç film önerilsin? (Top-N)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="İlk kaç filmi listeleyelim?"
            )

            # varsayılan diğerlerinin parametreleri
            top_n_item_based = 5
            selected_movie_title = None
            top_n_content = 5

        elif rec_type.startswith("Item-Based"):
            st.markdown("#### 🎬 Item-Based Parametreleri")

            top_n_item_based = st.slider(
                "Kaç benzer film gösterilsin?",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Referans filme (kullanıcının en son 5⭐ verdiği film) en çok benzeyen ilk N filmi göster."
            )

            # varsayılanlar
            min_overlap_ratio_pct = 60
            corr_threshold = 0.65
            max_neighbors = 7
            weighted_score_threshold = 3.5
            top_n_user_based = 5

            selected_movie_title = None
            top_n_content = 5

        else:
            st.markdown("#### 🏷️ Content-Based Parametreleri")

            # Çok büyük liste streaming sırasında ağır gelebilir,
            # istersen burada movie listesini popüler/top-rated 500 film ile daraltabilirsin.
            movie_titles_sorted = sorted(movie['title'].tolist())

            selected_movie_title = st.selectbox(
                "Referans Film Seçin",
                options=movie_titles_sorted,
                help="Bu filme tür olarak en çok benzeyen filmleri bulacağız."
            )

            top_n_content = st.slider(
                "Kaç benzer film gösterilsin?",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="En benzer ilk N filmi göster."
            )

            # varsayılanlar
            min_overlap_ratio_pct = 60
            corr_threshold = 0.65
            max_neighbors = 7
            weighted_score_threshold = 3.5
            top_n_user_based = 5

            top_n_item_based = 5

        run_button = st.button("🎬 Önerileri Hesapla", type="primary")

    # ---------------------------------
    # SAĞ KOLON: SONUÇLAR
    # ---------------------------------
    with right_col:
        st.markdown("### Çözüm Çıktısı")

        if not run_button:
            st.info("👈 Parametreleri seç ve '🎬 Önerileri Hesapla' butonuna bas.")
        else:
            # USER-BASED
            if rec_type.startswith("User-Based"):
                with st.spinner("User-Based hesaplanıyor..."):
                    pre_u = precompute_for_user_userbased(chosen_user)
                    result_user = finalize_user_based_from_cache(
                        precomputed=pre_u,
                        min_overlap_ratio_pct=min_overlap_ratio_pct,
                        corr_threshold=corr_threshold,
                        max_neighbors=max_neighbors,
                        weighted_score_threshold=weighted_score_threshold,
                        top_n=top_n_user_based,
                        chosen_user=chosen_user
                    )

                    status = result_user["status"]
                    debug_info = result_user.get("debug_info", {})

                    if status != "ok":
                        st.warning(f"⚠️ User-Based öneri üretilemedi. Durum: {status}")

                        # DEBUG GÖSTER: neden yok?
                        with st.expander("🔎 Debug: Ara Aşamalar (neden öneri yok?)"):
                            st.write("Aday kullanıcılar (candidate_users_df):")
                            st.dataframe(result_user.get("dbg_candidate_users_df", pd.DataFrame()).head(20))

                            st.write("Korelasyon tablosu (corr_df):")
                            st.dataframe(result_user.get("dbg_corr_df", pd.DataFrame()).head(20))

                            st.write("Filtre sonrası komşular (corr_filtered):")
                            st.dataframe(result_user.get("dbg_corr_filtered", pd.DataFrame()).head(20))

                            st.write("Komşu puanları (neighbor_ratings):")
                            st.dataframe(result_user.get("dbg_neighbor_ratings", pd.DataFrame()).head(20))

                    else:
                        recs_df = result_user["recommendations"]

                        if recs_df.empty:
                            st.info("Parametrelerle eşleşen öneri bulunamadı.")
                        else:
                            st.success("✅ User-Based önerileriniz hazır!")

                            st.markdown(
                                f"<div class='metric-card'>"
                                f"<div class='metric-title'>💬 Yorum</div>"
                                f"<div class='metric-value'>"
                                f"Benzer zevke sahip kullanıcıların sevdiği, benim henüz izlemediğim filmler."
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                            # Debug metrikler
                            with st.expander("🔍 Hesaplama Detayları (Debug / Görev Mantığı)"):
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("İzlenen Film", debug_info.get("movies_watched", "?"))
                                with col_b:
                                    st.metric("Aday Kullanıcı", debug_info.get("candidate_users", "?"))
                                with col_c:
                                    st.metric("Overlap+Corr Sonrası", debug_info.get("after_corr_users", "?"))
                                with col_d:
                                    st.metric("Kullanılan Komşu", debug_info.get("used_neighbors", "?"))

                            st.markdown(f"**📊 Toplam {len(recs_df)} öneri bulundu.**")
                            st.dataframe(
                                recs_df.head(20).reset_index(drop=True),
                                use_container_width=True
                            )

                        # Ayrıca başarılı durumda bile (öneri varsa bile) ara aşamaları göstermek isteyebilirsin:
                        with st.expander("🔎 Debug: Ara Aşamalar"):
                            st.write("Aday kullanıcılar (candidate_users_df):")
                            st.dataframe(result_user.get("dbg_candidate_users_df", pd.DataFrame()).head(20))

                            st.write("Korelasyon tablosu (corr_df):")
                            st.dataframe(result_user.get("dbg_corr_df", pd.DataFrame()).head(20))

                            st.write("Filtre sonrası komşular (corr_filtered):")
                            st.dataframe(result_user.get("dbg_corr_filtered", pd.DataFrame()).head(20))

                            st.write("Komşu puanları (neighbor_ratings):")
                            st.dataframe(result_user.get("dbg_neighbor_ratings", pd.DataFrame()).head(20))


            # ITEM-BASED
            elif rec_type.startswith("Item-Based"):
                with st.spinner("Item-Based hesaplanıyor..."):
                    pre_i = precompute_for_user_itembased(chosen_user)
                    status, ref_movie, sim_df = finalize_item_based_from_cache(
                        pre_i,
                        top_n_item_based
                    )

                if status != "ok":
                    if status == "no_five_star":
                        st.warning("⚠️ Bu kullanıcı hiç 5 puan vermemiş.")
                    elif status == "not_in_matrix":
                        st.warning(
                            f"⚠️ Referans film kullanıcı-film matrisinde yok: {pre_i['reference_movie']}"
                        )
                    else:
                        st.warning(f"⚠️ Item-Based öneri üretilemedi. Durum: {status}")
                else:
                    st.success("✅ Item-Based önerileriniz hazır!")

                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>🎯 Referans Film (Kullanıcının en son 5⭐ verdiği)</div>"
                        f"<div class='metric-value'>{ref_movie}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "Bu yaklaşım 'Bu filmi sevenler bunları da sevdi' mantığında çalışır. "
                        "Filmler arasındaki benzerliği kullanıcıların oy davranışına göre ölçer."
                    )

                    st.markdown(f"**📊 Toplam {len(sim_df)} benzer film bulundu.**")
                    st.dataframe(
                        sim_df.head(top_n_item_based).reset_index(drop=True),
                        use_container_width=True
                    )

            # CONTENT-BASED
            else:
                with st.spinner("Content-Based hesaplanıyor..."):
                    cb_result = content_based_recommender_cached(
                        movie_title=selected_movie_title,
                        top_n=top_n_content
                    )

                status = cb_result["status"]

                if status != "ok":
                    st.warning(f"⚠️ Film bulunamadı: {selected_movie_title}")
                else:
                    ref_movie = cb_result["reference_movie"]
                    ref_genres = cb_result["reference_genres"]
                    rec_df = cb_result["recommendations"]

                    st.success("✅ Content-Based önerileriniz hazır!")

                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>🎯 Referans Film</div>"
                        f"<div class='metric-value'>{ref_movie}</div>"
                        f"<div class='metric-title'>Türler: {ref_genres}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    with st.expander("💡 Bonus Görev 3: Content-Based Nasıl Çalışıyor?"):
                        st.info(
                            f"""
                            Bu yöntem kullanıcı davranışına değil, içeriğin kendisine bakar.

                            1️⃣ Referans filmin türlerini aldık: `{ref_genres}`  
                            2️⃣ Her filmi tür (genre) vektörü olarak temsil ediyoruz  
                            3️⃣ Cosine Similarity ile 'bu film hangi filmlere tür olarak en yakın?' sorusunu soruyoruz  
                            4️⃣ En yüksek benzerliğe sahip {top_n_content} filmi getiriyoruz

                            Güçlü yön: Yeni kullanıcıda bile çalışır (cold start daha küçük).  
                            Zayıf yön: Benzer tat çevresinde dönebilir (filter bubble).
                            """
                        )

                    st.markdown(f"**📊 Toplam {len(rec_df)} benzer film bulundu.**")
                    st.dataframe(
                        rec_df.head(top_n_content).reset_index(drop=True),
                        use_container_width=True
                    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
    <p>🎓 MIUUL DSMLBC19 Bootcamp - Hybrid Recommender System Case Study</p>
    <p>💡 Parametrelerle oynayarak önerilerin nasıl değiştiğini canlı göster ve her yöntemin mantığını tartıştır.</p>
</div>
""", unsafe_allow_html=True)
