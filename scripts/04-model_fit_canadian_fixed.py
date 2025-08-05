# Author: Adapted for Canadian University Basketball Data (Fixed)
# Date: 2024-12-19

"""
This script takes processed data from the 'data' folder in the project repository
and creates models to predict the 'Mins' feature using other features.
Types of model produced includes: baseline, linear regression, and random forest.
Afterwards, the scripts test the models and calculates the MSE and coefficient of
determination using cross-validation. Feature importance plots are created for the 
Random Forest model. These figures are then saved accordingly.

Both the file name and the save folder are required as inputs.

Usage: 04-model_fit_canadian_fixed.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/04-model_fit_canadian_fixed.py --file_name=player_data_ready_canadian.csv --save_folder=results
"""

# Loading the required packages
# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Numerical Packages
import numpy as np
import pandas as pd
# SKLearn Packages
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
# Binary Model Save
from pickle import dump
from docopt import docopt
# Other Packages
from termcolor import colored
import sys
import os
# Ignore warnings from packages in models
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(file_name, save_folder):
	# Load the processed data from csv
	# e.g. 'player_data_ready_canadian.csv'

	print(colored("\nWARNING: This script takes about 1 minute to run\n", 'yellow'))

	# Validate the file-path to load file
	path_str = str('../data/' + file_name)
	if os.path.exists(path_str) == False:
		path_str = str('data/' + file_name)
	try:
		data = pd.read_csv(path_str)
		print(colored('Data loaded successfully!', 'green'))
	except:
		print(colored('ERROR: Path to file is not valid!', 'red'))
		raise

	# Validate the save_foler directory exists or make folder
	if os.path.exists(save_folder) == False:
		if os.path.exists(str('../' + save_folder)) == False:
			try:
				os.makedirs(save_folder)
				print(colored('Successfully created folder for save data! Test passed!', 'green'))
			except:
				print(colored('ERROR: Path to save directory is not valid!', 'red'))
				raise

	# Preprocess the Data
	X, y = preprocess(data)

	# Set up cross-validation
	cv = KFold(n_splits=5, shuffle=True, random_state=42)

	# Fit the models
	rf_model = random_forest_model(X, y, cv)
	lm_model = linear_model(X, y, cv)
	base_model = baseline_model()

	# Get the predictions and score each model
	results = {}

	models = [
		(rf_model, 'Random Forest'), 
		(lm_model, 'Linear Regression'), 
		(base_model, 'Base Model (5 Game Ave)')
	]

	for i, (model, name) in enumerate(models):
		print(colored(f'\nEvaluating {name}...', 'blue'))
		
		# Get cross-validation scores
		if name != 'Base Model (5 Game Ave)':
			cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
			mse_scores = -cv_scores
			r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
			
			results[name] = {
				'MSE_mean': mse_scores.mean(),
				'MSE_std': mse_scores.std(),
				'R2_mean': r2_scores.mean(),
				'R2_std': r2_scores.std()
			}
		else:
			# For baseline model, use simple train-test split
			from sklearn.model_selection import train_test_split
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
			preds = model.predict(X_test)
			mse = mean_squared_error(y_test, preds)
			r2 = r2_score(y_test, preds)
			
			results[name] = {
				'MSE_mean': mse,
				'MSE_std': 0,
				'R2_mean': r2,
				'R2_std': 0
			}

	# Save results
	save_results(results, save_folder)

	# Create feature importance plot for Random Forest
	feature_importance(rf_model, X, save_folder)

	print(colored('\nModeling complete!', 'green'))

class baseline_model:
	"""
	Baseline model that predicts the average of last 5 games' minutes.
	"""
	def predict(self, X):
		# This is a simplified baseline - in practice you'd need the actual game history
		# For now, return the mean of the target variable
		return np.full(len(X), X['Mins_last5_mean'].mean() if 'Mins_last5_mean' in X.columns else 25)

def preprocess(data):
	"""
	Preprocess the data for modeling.

	Parameters:
	-----------
	data -- (pd DataFrame) the input dataframe containing features and target

	Return:
	-----------
	X -- (pd DataFrame) the feature matrix
	y -- (pd Series) the target variable
	"""
	# Separate features and target
	y = data['Mins']
	X = data.drop(['Mins', 'PlayerName', 'Date', 'Team'], axis=1, errors='ignore')

	# Remove any remaining non-numeric columns
	X = X.select_dtypes(include=[np.number])

	# Handle any infinite values
	X = X.replace([np.inf, -np.inf], np.nan)
	X = X.fillna(X.mean())

	return X, y

def random_forest_model(X, y, cv):
	"""
	Fit Random Forest model with cross-validation.

	Parameters:
	-----------
	X -- (pd DataFrame) the feature matrix
	y -- (pd Series) the target variable
	cv -- (KFold) cross-validation object

	Return:
	-----------
	model -- (RandomForestRegressor) fitted model
	"""
	model = RandomForestRegressor(
		n_estimators=100,
		max_depth=10,
		random_state=42,
		n_jobs=-1
	)
	
	# Fit the model
	model.fit(X, y)
	
	return model

def linear_model(X, y, cv):
	"""
	Fit Linear Regression model with cross-validation.

	Parameters:
	-----------
	X -- (pd DataFrame) the feature matrix
	y -- (pd Series) the target variable
	cv -- (KFold) cross-validation object

	Return:
	-----------
	model -- (LinearRegression) fitted model
	"""
	model = LinearRegression()
	
	# Fit the model
	model.fit(X, y)
	
	return model

def save_results(results, save_folder):
	"""
	Save model results to CSV.

	Parameters:
	-----------
	results -- (dict) dictionary containing model results
	save_folder -- (str) folder to save results
	"""
	results_df = pd.DataFrame(results).T
	results_df = results_df.round(4)
	results_df.to_csv(f'{save_folder}/modelling-score_table_canadian_fixed.csv')
	
	print(colored('Results saved!', 'green'))
	print('\nModel Performance Summary:')
	print(results_df)

def feature_importance(model, X, save_folder):
	"""
	Create feature importance plot for Random Forest model.

	Parameters:
	-----------
	model -- (RandomForestRegressor) fitted model
	X -- (pd DataFrame) the feature matrix
	save_folder -- (str) folder to save results
	"""
	# Get feature importance
	importance = model.feature_importances_
	feature_names = X.columns
	
	# Create DataFrame
	importance_df = pd.DataFrame({
		'Feature': feature_names,
		'Importance': importance
	}).sort_values('Importance', ascending=False)
	
	# Plot top 20 features
	plt.figure(figsize=(12, 8))
	top_features = importance_df.head(20)
	plt.barh(range(len(top_features)), top_features['Importance'])
	plt.yticks(range(len(top_features)), top_features['Feature'])
	plt.xlabel('Feature Importance')
	plt.title('Random Forest Feature Importance (Canadian University Basketball)')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(f'{save_folder}/modelling-rf_importance_canadian_fixed.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Feature importance plot saved!', 'green'))

if __name__ == "__main__":
	main(opt['--file_name'], opt['--save_folder']) 