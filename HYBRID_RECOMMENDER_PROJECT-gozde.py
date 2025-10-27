
#############################################
# PROJE: Hybrid Recommender System
#############################################

#############################################
# İş Problemi
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.


#############################################
# Veri Seti Hikayesi
#############################################

# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır. 27.278 filmde 20.000.263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.


#############################################
# Proje Görevleri
#############################################

# User Based Recommendation

#############################################
# GÖREV 1: Verinin Hazırlanması
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', 50)
pd.pandas.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import pickle

#############################################
# VERİ YÜKLEME (HIZLI VERSİYON)
#############################################

# Daha önce prepare_data.py ile oluşturduğumuz hazır veri dosyası.
# Bu dosyanın adı: prepare_data.pkl
# Bu dosyanın içinde şunlar var:
#   - movie            : film bilgileri (movieId, title, genres ...)
#   - rating           : kullanıcı-film puanları (userId, movieId, rating, timestamp)
#   - df_full          : movie + rating merge edilmiş tam tablo (userId, title, rating ...)
#   - common_movies    : yeterince rating almış popüler filmler listesi
#   - user_movie_df    : user x movie matrix (pivot edilmiş kullanıcı-film puan matrisi)
#
# Tekrar CSV okuma / merge / pivot yapmak istemiyoruz.Çünkü 20M satır bunu canlıda kilitler.Onun yerine bu pickle'dan RAM'e direkt alıyoruz.

with open("data/prepare_data.pkl", "rb") as f:
    data = pickle.load(f)

movie = data["movie"]
rating = data["rating"]
df = data["df_full"]
user_movie_df = data["user_movie_df"]

print("✅ Veri yüklendi.")
print("movie shape:", movie.shape)
print("rating shape:", rating.shape)
print("df_full shape:", df.shape)
print("user_movie_df shape:", user_movie_df.shape)

# ADIM 1: Movie ve Rating veri setlerini okutunuz.

# movie = pd.read_csv('.../movie.csv')
# rating = pd.read_csv('.../rating.csv')
# df = pd.merge(rating, movie, on="movieId")
# user_movie_df = df.pivot_table(index="userId", columns="title", values="rating")
# Bu hesap 20 milyon satırda çok uzun sürüyor. O yüzden biz bu ara çıktıları
# prepare_data.py ile bir kere üretip pickle olarak kaydettik.



# movieId, film adı ve filmin tür bilgilerini içeren veri seti

# movie = pd.read_csv('miuulpythonProject/pythonProject/datasets/movie_lens_dataset/movie.csv') (pickle olarak kaydettim tekrar çalıştırmıyorum)
movie.head()
movie.shape
movie["title"].nunique()
movie["title"].value_counts()

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
# rating = pd.read_csv('miuulpythonProject/pythonProject/datasets/movie_lens_dataset/rating.csv') (pickle olarak kaydettim tekrar çalıştırmıyorum)
rating.head()
rating.shape
rating["userId"].nunique()
rating["userId"].value_counts()


# ADIM 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanarak ekleyiniz.

# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# ID'lere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

# df_ = movie.merge(rating, how="left", on="movieId") (pickle olarak kaydettim tekrar çalıştırmıyorum)
# df = df_.copy() (buna da gerek yok)
df = data["df_full"]


def check_df(dataframe, head=5):
    print("################ Shape ##############")
    print(dataframe.shape)
    print('################ Types ##############')
    print(dataframe.dtypes)
    print('################ Head ###############')
    print(dataframe.head(head))
    print('################ Tail ###############')
    print(dataframe.tail(head))
    print('################ NA ###############')
    print(dataframe.isnull().sum())
    print('################ Quantiles ###############')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# Adım 3: Her bir film için toplam kaç kişinin oy kullandığını hesaplayınız.Toplam oy kullanılma sayısı 1000'in altında olan filmleri veri setinden çıkarınız.

# Her bir film veri setinde kaç kez geçiyor?
comment_counts = pd.DataFrame(df["title"].value_counts()) # movieId de kullanılabilir.
comment_counts

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies'de tutuyoruz ve veri setinden çıkarıyoruz.

rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape # (17766015, 6)
df.shape # (20000797, 6)
len(rare_movies) # 1000'den az oy alan film sayısı (24103)
df["title"].nunique()  # Orijinal veri setindeki benzersiz film sayısı (27262)


# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak da ratinglerin bulunduğu dataframe için pivot table oluşturunuz.


# user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") (pickle olarak kaydettim tekrar çalıştırmıyorum)

user_movie_df.head()


# Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.

"""
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('data/movie.csv')
    rating = pd.read_csv('data/rating.csv')
    df_ = movie.merge(rating, how="left", on="movieId")
    df = df_.copy()
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
"""

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.

# Rastgele bir userId seçelim:
#random_user = common_movies["userId"].sample(n=1, random_state=42).iloc[0]

random_user = 108170


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape # (1, 3159)

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list() # notna: bu sütunda en az bir izlenen film var mı?
len(movies_watched) # 186


#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve ID'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.

movies_watched_df = user_movie_df[movies_watched] # fancy index kullanarak filtreledik. random user'ın izlediği filmler ve diğer kullanıcıların old. bir dataframe oluşturduk
movies_watched_df.head()
movies_watched_df.shape # (138493, 186)
user_movie_df.shape # (138493, 3159)

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.


user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı ID'lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched) * 60 / 100 # 111.6
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies) # 2326

df["userId"].nunique() # 138493 kullanıcıdan 2326'ya indirgdik.

#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape # (2326, 186)

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]



# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape # (7, 2)
top_users.head()

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()


#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Adım 2: Film ID’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız. İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.

recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Adım 4:  movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Item-Based Recommendation
#############################################

# Görev 1: Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.

# random_user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.

# movie = pd.read_csv('miuulpythonProject/pythonProject/17_mentor_case/movie.csv')
# rating = pd.read_csv('miuulpythonProject/pythonProject/17_mentor_case/rating.csv')

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

movie[movie["movieId"] == movie_id]["title"].values[0] # 'Wild at Heart (1990)'
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Son iki adımı uygulayan fonksiyon

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Adım 5: Seçili film’in kendisi haricinde ilk 5 filmi öneri olarak veriniz.

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık. Wild at Heart (1990)
movies_from_item_based[1:6].index

# Index(['My Science Project (1985)', 'Mediterraneo (1991)', 'Old Man and the Sea, The (1958)', 'National Lampoon's Senior Trip (1995)', 'Clockwatchers (1997)'], dtype='object', name='title')





