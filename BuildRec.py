from surprise import SVD
from surprise.model_selection import cross_validate

# Use the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
trainset = surprise_data.build_full_trainset()
algo.fit(trainset)

# Evaluate the algorithm on the testset
cross_validate(algo, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)