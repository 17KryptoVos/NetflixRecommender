import pandas as pd
import numpy as np
import time
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


def main():
    # Load dataset
    df = pd.read_csv('input/combined_data_1.txt', names=['Cust-Id', 'Ratings'], usecols=[0, 1],
                     header=None)
    df.index = np.arange(0, len(df))

    # df_nan returns df with rows index that contain nan values
    df_nan = pd.DataFrame(pd.isnull(df.Ratings))
    df_nan = df_nan[df_nan['Ratings'] == True]
    # When reset_index is used, the old index becomes values in a column while the new index is sequential
    df_nan = df_nan.reset_index()

    # Numpy array
    movie_np = []
    movie_id = 1
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):  # excludes 23057834 in df_na
        temp = np.full((1, i - j - 1),
                       movie_id)
        # i-j-1 because you want to know the number of rows in between 0 and 548.
        # The number of rows between 0 and 548 correspond to the number of customer ratings for movie 1
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    # len(df) is the last customer rating for movie 4499 and df_nan.iloc[-1,0] is first row for customer ratings for 4499
    movie_np = np.append(movie_np, last_record)

    # Adjust dataframe with notnull and datatype
    df = df[pd.notnull(df['Ratings'])]
    df['Movie_Id'] = movie_np.astype(int)

    f = ['count', 'mean']

    # Benchmark movies
    df_movie_summary = df.groupby('Movie_Id')['Ratings'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
    movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
    print(f'Movie minimum times of review: {movie_benchmark}')

    # Benchmark users
    df_customer_summary = df.groupby('Cust-Id')['Ratings'].agg(f)
    customer_benchmark = round(df_customer_summary['count'].quantile(0.7), 0)
    customer_list = df_customer_summary[df_customer_summary['count'] < customer_benchmark].index
    print(f'Customer minimum times of review: {customer_benchmark}')

    # Slice df with benchmarked customer_list and movie_list
    df = df[~df['Movie_Id'].isin(movie_list)]
    df = df[~df['Cust-Id'].isin(customer_list)]
    df = df.reset_index(drop=True)

    # Create pivot table
    # df_p = pd.pivot_table(df, values='Ratings', index='Cust-Id', columns='Movie_Id')

    # Load movie titles into dataframe
    df_title = pd.read_csv('input\\movie_titles.csv', encoding="ISO-8859-1", names=['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace=True)

    # Count which user rates the most movies
    # df_count = df_p.count(axis='columns')
    # df_count = df_count.sort_values(ascending=False)
    # print(df_count)

    # Top 100K rows for faster evaluating
    reader = Reader()
    data = Dataset.load_from_df(df[['Cust-Id', 'Movie_Id', 'Ratings']][:100000], reader)

    # Choose algorithm
    algorithm = SVD()

    # Evaluate chosen algorithm
    cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # Viewing 5-star rated movies by chosen user
    df_chosen_user = df[(df['Cust-Id'] == '785314') & (df['Ratings'] == 5)]
    df_chosen_user = df_chosen_user.join(df_title)

    # Drop all ready seen movies from possibilities
    chosen_user = df_title.copy()
    chosen_user = chosen_user.reset_index()
    chosen_user = chosen_user[~chosen_user['Movie_Id'].isin(movie_list)]
    cond = chosen_user['Movie_Id'].isin(df_chosen_user['Movie_Id'])
    chosen_user.drop(chosen_user[cond].index, inplace=True)

    # Load complete dataset
    data = Dataset.load_from_df(df[['Cust-Id', 'Movie_Id', 'Ratings']], reader)

    # Create trainset
    trainset = data.build_full_trainset()

    # Fit algorithm
    algorithm.fit(trainset)

    # Predict
    chosen_user['Estimate_Score'] = chosen_user['Movie_Id'].apply(lambda x: algorithm.predict(785314, x).est)

    # Sort and clean prediction to print on console
    chosen_user = chosen_user.sort_values(['Estimate_Score'], ascending=False)
    chosen_user["Year"] = chosen_user["Year"].fillna(0.0).astype(int)
    print(chosen_user.head(n=10).to_string(index=False))

    # End timer
    print(f"Total prediction time {int(time.perf_counter())} seconds")

    # Print complete results to csv
    chosen_user.to_csv("output\\recommendation_results.csv", index=False)


if __name__ == "__main__":
    main()
