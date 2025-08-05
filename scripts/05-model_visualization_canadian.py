# Author: Model Visualization for Canadian University Basketball Data
# Date: 2024-12-19

"""
This script creates visualizations to compare model performance and show prediction accuracy.
It generates scatter plots, residual plots, and performance comparisons for all models.

Usage: 05-model_visualization_canadian.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/05-model_visualization_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results
"""

# Loading the required packages
# Models
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Numerical Packages
import numpy as np
import pandas as pd
# SKLearn Packages
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
# Other Packages
from docopt import docopt
from termcolor import colored
import sys
import os
# Ignore warnings from packages in models
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(file_name, save_folder):
	# Load the processed data from csv
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
				print(colored('Successfully created folder for save data!', 'green'))
			except:
				print(colored('ERROR: Path to save directory is not valid!', 'red'))
				raise

	# Preprocess the Data
	X, y = preprocess(data)

	# Split data for visualization
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train models
	models = {
		'Linear Regression': LinearRegression(),
		'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
		'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1)
	}

	# Train and predict
	results = {}
	predictions = {}
	
	for name, model in models.items():
		print(colored(f'\nTraining {name}...', 'blue'))
		model.fit(X_train, y_train)
		pred = model.predict(X_test)
		predictions[name] = pred
		
		# Calculate metrics
		mse = mean_squared_error(y_test, pred)
		mae = mean_absolute_error(y_test, pred)
		r2 = r2_score(y_test, pred)
		
		results[name] = {
			'MSE': mse,
			'MAE': mae,
			'R²': r2,
			'RMSE': np.sqrt(mse)
		}

	# Create visualizations
	create_performance_comparison(results, save_folder)
	create_prediction_scatter_plots(y_test, predictions, save_folder)
	create_residual_plots(y_test, predictions, save_folder)
	create_feature_importance_comparison(models, X, save_folder)
	
	print(colored('\nVisualization complete!', 'green'))

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

def create_performance_comparison(results, save_folder):
	"""
	Create a performance comparison chart for all models.
	"""
	print(colored('\nCreating performance comparison...', 'blue'))
	
	# Create DataFrame for plotting
	df_results = pd.DataFrame(results).T
	
	# Create subplots
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Model Performance Comparison (Canadian University Basketball)', fontsize=16)
	
	# MSE comparison
	axes[0, 0].bar(df_results.index, df_results['MSE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
	axes[0, 0].set_title('Mean Squared Error (Lower is Better)')
	axes[0, 0].set_ylabel('MSE')
	axes[0, 0].tick_params(axis='x', rotation=45)
	
	# RMSE comparison
	axes[0, 1].bar(df_results.index, df_results['RMSE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
	axes[0, 1].set_title('Root Mean Squared Error (Lower is Better)')
	axes[0, 1].set_ylabel('RMSE')
	axes[0, 1].tick_params(axis='x', rotation=45)
	
	# MAE comparison
	axes[1, 0].bar(df_results.index, df_results['MAE'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
	axes[1, 0].set_title('Mean Absolute Error (Lower is Better)')
	axes[1, 0].set_ylabel('MAE')
	axes[1, 0].tick_params(axis='x', rotation=45)
	
	# R² comparison
	axes[1, 1].bar(df_results.index, df_results['R²'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
	axes[1, 1].set_title('R² Score (Higher is Better)')
	axes[1, 1].set_ylabel('R²')
	axes[1, 1].tick_params(axis='x', rotation=45)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/model_performance_comparison_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	# Save results to CSV
	df_results.to_csv(f'{save_folder}/model_performance_detailed_canadian.csv')
	
	print(colored('Performance comparison saved!', 'green'))
	
	# Print best model
	best_model = df_results['R²'].idxmax()
	best_r2 = df_results.loc[best_model, 'R²']
	print(f'\n🏆 Best Model: {best_model} (R² = {best_r2:.4f})')

def create_prediction_scatter_plots(y_test, predictions, save_folder):
	"""
	Create scatter plots showing predicted vs actual values for each model.
	"""
	print(colored('\nCreating prediction scatter plots...', 'blue'))
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle('Predicted vs Actual Minutes Played (Canadian University Basketball)', fontsize=16)
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
	
	for i, (name, pred) in enumerate(predictions.items()):
		axes[i].scatter(y_test, pred, alpha=0.6, color=colors[i])
		
		# Add perfect prediction line
		min_val = min(y_test.min(), pred.min())
		max_val = max(y_test.max(), pred.max())
		axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
		
		# Add R² to plot
		r2 = r2_score(y_test, pred)
		axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
					bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
		
		axes[i].set_xlabel('Actual Minutes')
		axes[i].set_ylabel('Predicted Minutes')
		axes[i].set_title(name)
		axes[i].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/prediction_scatter_plots_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Prediction scatter plots saved!', 'green'))

def create_residual_plots(y_test, predictions, save_folder):
	"""
	Create residual plots for each model.
	"""
	print(colored('\nCreating residual plots...', 'blue'))
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle('Residual Plots (Canadian University Basketball)', fontsize=16)
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
	
	for i, (name, pred) in enumerate(predictions.items()):
		residuals = y_test - pred
		
		axes[i].scatter(pred, residuals, alpha=0.6, color=colors[i])
		axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
		
		# Add residual statistics
		mean_residual = residuals.mean()
		std_residual = residuals.std()
		axes[i].text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}', 
					transform=axes[i].transAxes,
					bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
		
		axes[i].set_xlabel('Predicted Minutes')
		axes[i].set_ylabel('Residuals (Actual - Predicted)')
		axes[i].set_title(name)
		axes[i].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/residual_plots_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Residual plots saved!', 'green'))

def create_feature_importance_comparison(models, X, save_folder):
	"""
	Create feature importance comparison for tree-based models.
	"""
	print(colored('\nCreating feature importance comparison...', 'blue'))
	
	# Get feature importance for tree-based models
	importance_data = {}
	
	for name, model in models.items():
		if hasattr(model, 'feature_importances_'):
			importance_data[name] = model.feature_importances_
	
	if not importance_data:
		print(colored('No tree-based models found for feature importance comparison.', 'yellow'))
		return
	
	# Create comparison plot
	fig, axes = plt.subplots(1, len(importance_data), figsize=(6*len(importance_data), 8))
	if len(importance_data) == 1:
		axes = [axes]
	
	fig.suptitle('Feature Importance Comparison (Canadian University Basketball)', fontsize=16)
	
	for i, (name, importance) in enumerate(importance_data.items()):
		# Get top 15 features
		feature_importance_df = pd.DataFrame({
			'Feature': X.columns,
			'Importance': importance
		}).sort_values('Importance', ascending=False).head(15)
		
		axes[i].barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
					color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
		axes[i].set_yticks(range(len(feature_importance_df)))
		axes[i].set_yticklabels(feature_importance_df['Feature'])
		axes[i].set_xlabel('Importance')
		axes[i].set_title(f'{name} - Top 15 Features')
		axes[i].invert_yaxis()
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/feature_importance_comparison_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Feature importance comparison saved!', 'green'))

if __name__ == "__main__":
	main(opt['--file_name'], opt['--save_folder']) 