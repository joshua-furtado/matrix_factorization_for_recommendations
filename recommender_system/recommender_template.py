import numpy as np
import pandas as pd
import sys  # can use sys to take command line arguments
import recommender_functions as rf


class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''

    def __init__(self, ):
        pass

    def fit(self, reviews_path, movies_path, latent_features=12, learning_rate=0.0001, iters=100):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions

        INPUT:
        reviews_path - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_path - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None
        '''

        self.reviews = pd.read_csv(reviews_path)
        self.movies = pd.read_csv(movies_path)

        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        train_user_item = self.reviews[[
            'user_id', 'movie_id', 'rating', 'timestamp']]
        self.train_data_df = train_user_item.groupby(['user_id', 'movie_id'])[
            'rating'].max().unstack()
        self.train_data_np = np.array(self.train_data_df)

        # SVD based fit
        self.FunkSVD()

        # knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)

    def predict_rating(self, user_id, movie_id):
        '''
        makes predictions of a rating for a user on a movie-user combo

        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        prediction - the predicted rating for user_id-movie_id according to FunkSVD
        '''

        self.user_ids_series = np.array(self.train_data_df.index)
        self.movie_ids_series = np.array(self.train_data_df.columns)

        try:
            # User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]
            print('ID 1')
            # Take dot product of that row and column in U and V to make prediction
            prediction = np.dot(
                self.user_mat[user_row, :], self.movie_mat[:, movie_col])
            print('ID 2')
            self.print_prediction_summary(user_id, movie_id, prediction)
            return prediction

        except:
            print("Sorry, the user id or movie id does not exist.")
            return None

    def print_prediction_summary(self, user_id, movie_id, prediction):
        '''
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df
        prediction - the predicted rating for user_id-movie_id

        OUTPUT:
        None - prints a statement about the user, movie, and prediction made

        '''

        movie_name = self.movies[self.movies['movie_id']
                                 == movie_id]['movie'].iloc[0]
        print(movie_name)
        print(
            f'For user {user_id}, we predict a rating of {np.round(prediction)} for movie: {movie_name}.')

    def make_recs(self, _id, _id_type='movie', rec_num=5):
        '''
        given a user id or a movie that an individual likes
        make recommendations

        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        rec_ids - (array) a list or numpy array of recommended movies by id                  
        rec_names - (array) a list or numpy array of recommended movies by name
        '''

        if _id_type == 'user':  # recommend movies to a user
            # check if user id is present in training data
            if _id in self.user_ids_series:
                index = np.where(self.train_data_df.index == _id)[0][0]
                # get predicted movie ratings for the user
                predicted_ratings = np.dot(
                    self.user_mat[index, :], self.movie_mat)

                # get array indices corresponding to top movies
                indices = predicted_ratings.argsort()[-rec_num:][::-1]
                rec_ids = list(self.train_data_df.columns[indices])
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # use ranked movies
                rec_names = rf.popular_recommendations(
                    _id, rec_num, self.ranked_movies)
                rec_ids = list(self.ranked_movies.index[:rec_num])

        else:  # recommend movies similar to a movie
            if _id in self.movie_ids_series:
                rec_names = rf.find_similar_movies(_id, self.movies)[:rec_num]
                rec_ids = list(
                    self.movies[self.movies['movie'].isin(rec_names)]['movie_id'])
            else:
                print("Sorry, this movie id does not exist in our system.")
                rec_ids, rec_names = None, None

        return rec_ids, rec_names

    def FunkSVD(self):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization
        '''

        # Set up useful values to be used through the rest of the function
        self.n_users = self.train_data_np.shape[0]
        self.n_movies = self.train_data_np.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.train_data_np))

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if the rating exists
                    if self.train_data_np[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.train_data_np[i, j] - \
                            np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * \
                                (2*diff*movie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * \
                                (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))

        self.user_mat = user_mat
        self.movie_mat = movie_mat


if __name__ == '__main__':
    # test different parts to make sure it works
    import recommender_template as r
    divide = '\n---------------------------------------------------------------\n'

    # instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_path='data/train_data.csv',
            movies_path='data/movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recs(8, 'user'), divide)  # user in the dataset
    print(rec.make_recs(1, 'user'), divide)  # user not in dataset
    print(rec.make_recs(1853728), divide)  # movie in the dataset
    print(rec.make_recs(1), divide)  # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)