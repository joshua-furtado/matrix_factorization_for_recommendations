# Matrix Factorization for Recommendations

## Description

**matrix_factorization_for_recommendations** uses SVD to find latent features related to the movies and customers, and use these to recommend movies.

## Dependencies

matrix_factorization_for_recommendations requires:

- NumPy
- pandas

## Folder Descriptions

1. [notebooks](https://github.com/joshua-furtado/matrix_factorization_for_recommendations/tree/main/notebooks)

	- Contains the following Jupyter notebooks:  
		- 1_Intro_to_SVD.ipynb: Use NumPy's built-in SVD functionality
		- 2_Implementing_FunkSVD.ipynb: Implement gradient descent based FunkSVD
		- 3_How_Are_We_Doing.ipynb: Fit FunkSVD on training data and test on validation data
		- 4_Cold_Start_Problem.ipynb: Fix the cold start problem by blending in other recommendation system techniques such as knowledge and content based.

2. [recommender_system](https://github.com/joshua-furtado/matrix_factorization_for_recommendations/tree/main/recommender_system)

	- Puts the work done in the notebooks together to build a Recommender() class
	- Sample code for Python shell:
		```bash
		>>> from recommender_template import Recommender
		>>> rec = Recommender()
		>>> rec.fit(reviews_path='data/train_data.csv', movies_path='data/movies_clean.csv', learning_rate=.01, iters=1)
		>>> rec.predict_rating(user_id=8, movie_id=2844)
		>>> print(rec.make_recs(8, 'user'))
		```

## Authors

**Joshua Furtado**
