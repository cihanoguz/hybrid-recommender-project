
#############################################
# PROJECT: Hybrid Recommender System
#############################################

#############################################
# Business Problem
#############################################

# Make 10 film recommendations for a given user ID using item-based and user-based recommender methods.


#############################################
# Dataset Story
#############################################

# The dataset is provided by MovieLens, a film recommendation service. It contains films along with rating scores given to these films. It contains 20,000,263 ratings on 27,278 films. This dataset was created on October 17, 2016. It contains data from 138,493 users between January 9, 1995 and March 31, 2015. Users were randomly selected. It is known that all selected users have rated at least 20 films.


#############################################
# Project Tasks
#############################################

# User Based Recommendation

#############################################
# TASK 1: Data Preparation
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', 50)
pd.pandas.set_option('display.width', 500)

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import pickle

#############################################
# DATA LOADING (QUICK VERSION)
#############################################

# Previously created data file using prepare_data.py.
# This file's name: prepare_data.pkl
# This file contains:
#   - movie            : film information (movieId, title, genres ...)
#   - rating           : user-film ratings (userId, movieId, rating, timestamp)
#   - df_full          : complete table with movie + rating merged (userId, title, rating ...)
#   - common_movies    : list of popular films that received enough ratings
#   - user_movie_df    : user x movie matrix (pivoted user-film rating matrix)
#
# We don't want to read CSV / merge / pivot again. Because 20M rows will lock it up in production. Instead, we load it directly from this pickle into RAM.

with open("data/prepare_data.pkl", "rb") as f:
    data = pickle.load(f)

movie = data["movie"]
rating = data["rating"]
df = data["df_full"]
user_movie_df = data["user_movie_df"]

print("✅ Data loaded.")
print("movie shape:", movie.shape)
print("rating shape:", rating.shape)
print("df_full shape:", df.shape)
print("user_movie_df shape:", user_movie_df.shape)

# STEP 1: Load Movie and Rating datasets.

# movie = pd.read_csv('.../movie.csv')
# rating = pd.read_csv('.../rating.csv')
# df = pd.merge(rating, movie, on="movieId")
# user_movie_df = df.pivot_table(index="userId", columns="title", values="rating")
# This calculation takes too long on 20 million rows. That's why we generate these intermediate outputs
# once with prepare_data.py and save them as pickle.



# Dataset containing movieId, film name and film genre information

# movie = pd.read_csv('miuulpythonProject/pythonProject/datasets/movie_lens_dataset/movie.csv') (saved as pickle, not running again)
movie.head()
movie.shape
movie["title"].nunique()
movie["title"].value_counts()

# Dataset containing UserID, film name, rating given to film and time information
# rating = pd.read_csv('miuulpythonProject/pythonProject/datasets/movie_lens_dataset/rating.csv') (saved as pickle, not running again)
rating.head()
rating.shape
rating["userId"].nunique()
rating["userId"].value_counts()


# STEP 2: Add film names and genres to the Rating dataset using the movie dataset.

# In Rating, users' rated films only have IDs.
# We add film names and genres corresponding to IDs from the movie dataset.

# df_ = movie.merge(rating, how="left", on="movieId") (saved as pickle, not running again)
# df = df_.copy() (no need for this either)
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


# Step 3: Calculate how many people rated each film. Remove films from the dataset that have less than 1000 total ratings.

# How many times does each film appear in the dataset?
comment_counts = pd.DataFrame(df["title"].value_counts()) # movieId can also be used.
comment_counts
rating.head()
movie.head()
# We keep the names of films with less than 1000 total ratings in rare_movies and remove them from the dataset.

rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape # (17766015, 6)
df.shape # (20000797, 6)
len(rare_movies) # number of films with less than 1000 ratings (24103)
df["title"].nunique()  # unique film count in original dataset (27262)


# Step 4: Create a pivot table for a dataframe where userIDs are in the index, film names are in columns, and ratings are the values.


# user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating") (saved as pickle, not running again)

user_movie_df.head()


# Step 5: Functionize all operations performed.

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
# Task 2: Determining Films Watched by User for Recommendations
#############################################

# Step 1: Select a random user id.

# Let's select a random userId:
#random_user = common_movies["userId"].sample(n=1, random_state=42).iloc[0]

random_user = 108170


# Step 2: Create a new dataframe named random_user_df consisting of observation units belonging to the selected user.

random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape # (1, 3159)

# Step 3: Assign films that the selected user has rated to a list named movies_watched.

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list() # notna: is there at least one watched film in this column?
len(movies_watched) # 186


#############################################
# Task 3: Accessing Data and IDs of Other Users Who Watched the Same Films
#############################################

# Step 1: Select columns belonging to films watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.

movies_watched_df = user_movie_df[movies_watched] # we filtered using fancy indexing. we created a dataframe with films watched by random user and other users' data
movies_watched_df.head()
movies_watched_df.shape # (138493, 186)
user_movie_df.shape # (138493, 3159)

# Step 2: Create a new dataframe named user_movie_count containing information on how many of the films watched by the selected user each user has watched.


user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# Step 3: Create a list named users_same_movies from user IDs of those who watched 60% or more of the films the selected user rated.

perc = len(movies_watched) * 60 / 100 # 111.6
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies) # 2326

df["userId"].nunique() # reduced from 138493 users to 2326.

#############################################
# Task 4: Determining Most Similar Users to the User for Recommendations
#############################################

# Step 1: Filter the movies_watched_df dataframe so that user IDs showing similarity with the selected user within the user_same_movies list will be found.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape # (2326, 186)

# Step 2: Create a new corr_df dataframe where correlations between users will be calculated.

corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df[corr_df["user_id_1"] == random_user]



# Step 3: Filter users with high correlation (above 0.65) with the selected user and create a new dataframe named top_users.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape # (7, 2)
top_users.head()

# Step 4: Merge top_users dataframe with rating dataset

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()


#############################################
# Task 5: Calculation of Weighted Average Recommendation Score and Keeping First 5 Films
#############################################

# Step 1: Create a new variable named weighted_rating consisting of the product of each user's corr and rating values.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.sort_values(by='corr', ascending=True)

# Step 2: Create a new dataframe named recommendation_df containing film ID and the average value of all users' weighted ratings for each film.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Step 3: Select films with weighted rating greater than 3.5 in recommendation_df and sort by weighted rating. Save the first 5 observations as movies_to_be_recommend.

recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Step 4: Get film names from the movie dataset and select the first 5 films to be recommended.

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Item-Based Recommendation
#############################################

# Task 1: Make item-based recommendations based on the name of the film the user last watched and gave the highest rating.

# random_user = 108170

# Step 1: Load movie and rating datasets.

# movie = pd.read_csv('miuulpythonProject/pythonProject/17_mentor_case/movie.csv')
# rating = pd.read_csv('miuulpythonProject/pythonProject/17_mentor_case/rating.csv')

# Step 2: Get the ID of the film with the most recent rating among the films the user for recommendations rated 5.

movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Step 3: Filter the user_movie_df dataframe created in the User based recommendation section according to the selected film id.

movie[movie["movieId"] == movie_id]["title"].values[0] # 'Wild at Heart (1990)'
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Step 4: Using the filtered dataframe, find the correlation between the selected film and other films and sort.

user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Function that applies the last two steps

def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Step 5: Give the first 5 films except the selected film itself as recommendations.

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)

# From 1 to 6. The film itself is at 0. We excluded that. Wild at Heart (1990)
movies_from_item_based[1:6].index

# Index(['My Science Project (1985)', 'Mediterraneo (1991)', 'Old Man and the Sea, The (1958)', 'National Lampoon's Senior Trip (1995)', 'Clockwatchers (1997)'], dtype='object', name='title')





#############################################
# BONUS: Content-Based Recommendation
#############################################

# OBJECTIVE:
# Recommend similar films based solely on films' content (especially 'genres' information), without relying on user behaviors (ratings).
# User-Based and Item-Based approaches rely on rating data.
# If the user is new / no one has rated the film yet, these two methods struggle. This is called the "cold start problem".
# Content-Based approach works with logic like "this film's taste is similar to this one". So it can recommend similar films using only content information.

# BASIC IDEA:
# 1. We convert each film's genre information (genres) into a numerical vector. We use TF-IDF for this.
# TF-IDF (Term Frequency - Inverse Document Frequency):
# Reduces the weight of very common genres (e.g., Drama).
# Considers rarer genres (e.g., Film-Noir) more valuable.
# 2. We measure similarity between films using cosine similarity.
# Cosine similarity:
# Measures the angle between two vectors.
# If close to 1, they are very similar, if close to 0, they are unrelated.
# 3. We return the first N films most similar to the selected film.

#############################################
# Required Libraries
#############################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#############################################
# Step 1: Preparing Genre Information
#############################################

# At this stage, we will use the 'genres' column in the 'movie' dataframe.
# Genre information may be NaN in some films. If NaN remains, TF-IDF will give an error. That's why we convert it to an empty string.

movie["genres"] = movie["genres"].fillna("")

#############################################
# Step 2: Creating TF-IDF Vectors
#############################################

# Extract a vector like a "genre profile" for each film.

# Example: "Action|Adventure|Sci-Fi" we split this string and treat it like a bag of words.
# token_pattern=r"[^|]+"
# Count each part split by "|" character as a "word".
# So "Action|Adventure|Sci-Fi" -> ["Action", "Adventure", "Sci-Fi"]

tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
tfidf_matrix = tfidf.fit_transform(movie["genres"])

print("TF-IDF matrix shape:", tfidf_matrix.shape) # (27278, 20)

# tfidf_matrix dimensions:
# rows -> films
# columns -> genre terms
# each cell -> that film's weight (TF-IDF score) for that genre

#############################################
# Step 3: Cosine Similarity Matrix
#############################################

# cosine_similarity(tfidf_matrix, tfidf_matrix)
# Compares all films to each other one by one.
# Returns a (N x N) dimensional matrix.
# -> cosine_sim_matrix[i, j] = similarity between film i and film j
# This matrix can be large (full MovieLens data is very large), but we already reduced the data in the project.

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Cosine similarity matrix shape:", cosine_sim_matrix.shape) # (27278, 27278)

# Now we can ask the question "films most similar to this film" using this matrix.

#############################################
# Step 4: Content-Based Recommendation Function
#############################################

# This function takes a film name (like "Matrix, The (1999)"), finds that film and returns the first N films most similar to it (excluding itself).

def content_based_recommender(title, movie_df, sim_matrix, top_n=5):
    """
    Content-Based Recommender
    -------------------------
    Returns the top_n films most similar to the given film title.

    PARAMETERS:
        title      : reference film name (string). Must appear exactly in movie_df["title"].
        movie_df   : dataframe containing film information (movie).
        sim_matrix : cosine similarity matrix (cosine_sim_matrix).
        top_n      : how many recommendations should we return?

    RETURNS:
        result_df  : recommended films and similarity scores (DataFrame)
    """

    # 1) Does the film really exist?
    if title not in movie_df["title"].values:
        print("⚠ Film not found:", title)
        return pd.DataFrame()

    # 2) Find the index of the relevant film
    idx = movie_df[movie_df["title"] == title].index[0]

    # 3) Get similarity scores of this film with all other films
    sim_scores = list(enumerate(sim_matrix[idx]))

    # 4) Sort scores from largest to smallest
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 5) First element is the film with itself (score = 1.0). We skip that.
    sim_scores = sim_scores[1: top_n + 1]

    # 6) Collect film information corresponding to the indices we selected
    film_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    recs = movie_df.iloc[film_indices][["title", "genres"]].copy()
    recs["similarity_score"] = [round(s, 3) for s in scores]

    # 7) Make column names more presentation-friendly
    recs = recs.rename(columns={
        "title": "Recommended Film",
        "genres": "Genres",
        "similarity_score": "Similarity Score"
    })

    return recs.reset_index(drop=True)


#############################################
# Step 5: Example Usage of Function
#############################################

# "What was the film the user last gave 5 to?" Let's also take the same film as reference here and make content-similar film recommendations.

reference_title = movie[movie["movieId"] == movie_id]["title"].values[0]
print("Reference Film (user's last 5★ rating):", reference_title)

content_based_suggestions = content_based_recommender(
    title=reference_title,
    movie_df=movie,
    sim_matrix=cosine_sim_matrix,
    top_n=5
)

print("\nContent-Based Recommendations:")
print(content_based_suggestions)



