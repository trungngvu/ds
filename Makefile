env:
	pipenv install -r requirements.txt

connect-env:
	pipenv shell

training-mlp:
	cd ./ml_model_build_deep_learning && python3 MLP_sklearn.py

training-dimension:
	cd ./ml_model_build_dimensionality_reduction && python3 dimensionality_reduction.py

training-knn:
	cd ./ml_model_build_nearest_neighbor && python3 nearest_neighbor_model_build.py

training-random-forest:
	cd ./ml_model_build_random_forest && python3 random_forest_model_build.py

test-random-forest:
	cd ./ml_model_build_random_forest && python3 random_forest_feature_testing.py

training-svm:
	cd ./ml_model_build_support_vector_machine && python3 support_vector_machine_model_build.py
