def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('movie.csv')
    rating = pd.read_csv('rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    df = df.iloc[:50000, :]
    df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
    df['title'] = df['title'].apply(lambda x: x.strip())
    a = pd.DataFrame(df["title"].value_counts())
    rare_movies = a[a["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def user_based_recommender():
    import pandas as pd
    random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
    random_user_df = user_movie_df[user_movie_df.index == random_user]

    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                          random_user_df[movies_watched]])
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
    temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
    temp.columns = ['sum_corr', 'sum_weighted_rating']

    recommendation_df = pd.DataFrame()
    recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
    recommendation_df['movieId'] = temp.index
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)

    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'])]

user_based_recommender()

