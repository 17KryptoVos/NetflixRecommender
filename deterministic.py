import json
import time
from collections import Counter
import random
import sys


def generate_random_user(data):
    sequence = list(data.keys())
    return random.choice(sequence)


def recommend_movie(chosen_user_id, data):
    count = 0
    common_liked_movies = {}
    chosen_user_list = []

    # Find user
    print(f"Recommending for user {chosen_user_id}")

    # Iterate movies liked by chosen user
    for movie_liked_by_user in data[chosen_user_id]:
        # Iterate all user_id's in the data
        chosen_user_list.append(movie_liked_by_user)
        for user_id in data:
            for movie in data[user_id]:
                # If other users like the movie from our chosen user, it is counted
                if movie == movie_liked_by_user:
                    count += 1
        # If movie count is more than 1, it is liked by other users and appended to the common liked movies dict.
        if count != 1:
            common_liked_movies.update({movie_liked_by_user: count})
        count = 0

    # Return the most common liked movie(s) next to current_movie with not viewed
    most_liked_movie_id = max(common_liked_movies, key=common_liked_movies.get)
    print(
        f"The most common-liked movie is movie {most_liked_movie_id} with {common_liked_movies[most_liked_movie_id]} ratings")
    print("Going to check all of the movies from all the people who liked this movie!")

    # Generate a list of movies based on the other users 5 stars ratings
    top_list = []
    for user_id in data:
        # For each user, we create a list of movies it liked
        individual_user_list_of_liked_movies = []
        for movie in data[user_id]:
            individual_user_list_of_liked_movies.append(movie)
        # Only add them if they have the most common liked movie in their list
        if most_liked_movie_id in individual_user_list_of_liked_movies:
            top_list.append(individual_user_list_of_liked_movies)

    # Remove all movies already seen by chosen user out of the list of possibilities
    reduced_list_of_movies_to_recommend = []
    for list_by_individual_user in top_list:
        temp_list = [item for item in list_by_individual_user if item not in chosen_user_list]
        if temp_list:
            reduced_list_of_movies_to_recommend.append(temp_list)

    # Count which movie has the highest number of 5-star ratings
    counter = Counter()
    for temp_list in reduced_list_of_movies_to_recommend:
        counter += Counter(temp_list)

    movie_id_to_recommend = max(counter, key=counter.get)
    print(
        f"The recommended movie is {movie_id_to_recommend} with a total of {counter[movie_id_to_recommend]} five star ratings")


def main():
    # Start timer
    time.perf_counter()

    # Load data (pipeline/cli)
    data = json.load(sys.stdin)

    # Choose a random user
    # recommendMovie(generateRandomUser(data), data)

    # Choose a specific user
    recommend_movie(str(785314), data)

    # End timer
    print(f"Total prediction time {int(time.perf_counter())} seconds")


if __name__ == "__main__":
    main()
