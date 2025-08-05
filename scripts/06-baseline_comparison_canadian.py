# Author: Baseline Comparison for Canadian University Basketball Data
# Date: 2024-12-19

"""
This script compares the Linear Regression model against simple rolling average baselines.
It tests 3-game and 5-game rolling averages to see if the ML model actually adds value.

Usage: 06-baseline_comparison_canadian.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/06-baseline_comparison_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results
"""

# Loading the required packages
# Models
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

	# Split data for comparison
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train Linear Regression model
	print(colored('\nTraining Linear Regression model...', 'blue'))
	lr_model = LinearRegression()
	lr_model.fit(X_train, y_train)
	lr_pred = lr_model.predict(X_test)

	# Calculate rolling averages for comparison
	print(colored('\nCalculating rolling average baselines...', 'blue'))
	rolling_3_pred = calculate_rolling_average_predictions(data, 3, X_test.index)
	rolling_5_pred = calculate_rolling_average_predictions(data, 5, X_test.index)

	# Compare all methods
	results = {}
	methods = {
		'Linear Regression': lr_pred,
		'3-Game Rolling Average': rolling_3_pred,
		'5-Game Rolling Average': rolling_5_pred
	}

	for name, predictions in methods.items():
		# Filter out NaN predictions (where we don't have enough history)
		valid_mask = ~np.isnan(predictions)
		if valid_mask.sum() > 0:
			valid_pred = predictions[valid_mask]
			valid_actual = y_test.iloc[valid_mask]
			
			mse = mean_squared_error(valid_actual, valid_pred)
			mae = mean_absolute_error(valid_actual, valid_pred)
			r2 = r2_score(valid_actual, valid_pred)
			rmse = np.sqrt(mse)
			
			results[name] = {
				'MSE': mse,
				'MAE': mae,
				'R²': r2,
				'RMSE': rmse,
				'Valid_Predictions': valid_mask.sum(),
				'Total_Predictions': len(predictions)
			}
		else:
			print(colored(f'Warning: No valid predictions for {name}', 'yellow'))

	# Create comparison visualizations
	create_baseline_comparison(results, save_folder)
	create_prediction_comparison_plots(y_test, methods, save_folder)
	create_error_distribution_plots(y_test, methods, save_folder)
	
	print(colored('\nBaseline comparison complete!', 'green'))
	
	# Print summary
	print_summary(results)

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

def calculate_rolling_average_predictions(data, window, test_indices):
	"""
	Calculate rolling average predictions for the test set.
	
	Parameters:
	-----------
	data -- (pd DataFrame) the full dataset
	window -- (int) the rolling window size
	test_indices -- (Index) indices of test set
	
	Return:
	-----------
	predictions -- (np.array) rolling average predictions
	"""
	predictions = np.full(len(test_indices), np.nan)
	
	# Group by player and calculate rolling averages
	for player in data['PlayerName'].unique():
		player_data = data[data['PlayerName'] == player].sort_values('Date')
		
		# Calculate rolling average
		rolling_avg = player_data['Mins'].rolling(window=window, min_periods=1).mean()
		
		# Find test indices for this player
		player_test_mask = data.loc[test_indices, 'PlayerName'] == player
		player_test_indices = test_indices[player_test_mask]
		
		if len(player_test_indices) > 0:
			# Map test indices to player data indices
			for test_idx in player_test_indices:
				player_idx = data.index.get_loc(test_idx)
				# Use the rolling average from the current position (safe indexing)
				if player_idx < len(rolling_avg):
					predictions[test_indices.get_loc(test_idx)] = rolling_avg.iloc[player_idx]
				else:
					# Fallback to the last available rolling average
					predictions[test_indices.get_loc(test_idx)] = rolling_avg.iloc[-1]
	
	return predictions

def create_baseline_comparison(results, save_folder):
	"""
	Create a comprehensive comparison chart for all methods.
	"""
	print(colored('\nCreating baseline comparison...', 'blue'))
	
	# Create DataFrame for plotting
	df_results = pd.DataFrame(results).T
	
	# Create subplots
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Linear Regression vs Rolling Averages (Canadian University Basketball)', fontsize=16)
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
	
	# MSE comparison
	axes[0, 0].bar(df_results.index, df_results['MSE'], color=colors[:len(df_results)])
	axes[0, 0].set_title('Mean Squared Error (Lower is Better)')
	axes[0, 0].set_ylabel('MSE')
	axes[0, 0].tick_params(axis='x', rotation=45)
	
	# RMSE comparison
	axes[0, 1].bar(df_results.index, df_results['RMSE'], color=colors[:len(df_results)])
	axes[0, 1].set_title('Root Mean Squared Error (Lower is Better)')
	axes[0, 1].set_ylabel('RMSE')
	axes[0, 1].tick_params(axis='x', rotation=45)
	
	# MAE comparison
	axes[1, 0].bar(df_results.index, df_results['MAE'], color=colors[:len(df_results)])
	axes[1, 0].set_title('Mean Absolute Error (Lower is Better)')
	axes[1, 0].set_ylabel('MAE')
	axes[1, 0].tick_params(axis='x', rotation=45)
	
	# R² comparison
	axes[1, 1].bar(df_results.index, df_results['R²'], color=colors[:len(df_results)])
	axes[1, 1].set_title('R² Score (Higher is Better)')
	axes[1, 1].set_ylabel('R²')
	axes[1, 1].tick_params(axis='x', rotation=45)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/baseline_comparison_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	# Save results to CSV
	df_results.to_csv(f'{save_folder}/baseline_comparison_detailed_canadian.csv')
	
	print(colored('Baseline comparison saved!', 'green'))

def create_prediction_comparison_plots(y_test, methods, save_folder):
	"""
	Create scatter plots comparing all prediction methods.
	"""
	print(colored('\nCreating prediction comparison plots...', 'blue'))
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle('Predicted vs Actual Minutes - All Methods (Canadian University Basketball)', fontsize=16)
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
	
	for i, (name, pred) in enumerate(methods.items()):
		# Filter out NaN predictions
		valid_mask = ~np.isnan(pred)
		if valid_mask.sum() > 0:
			valid_pred = pred[valid_mask]
			valid_actual = y_test.iloc[valid_mask]
			
			axes[i].scatter(valid_actual, valid_pred, alpha=0.6, color=colors[i])
			
			# Add perfect prediction line
			min_val = min(valid_actual.min(), valid_pred.min())
			max_val = max(valid_actual.max(), valid_pred.max())
			axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
			
			# Add R² to plot
			r2 = r2_score(valid_actual, valid_pred)
			axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nValid: {valid_mask.sum()}/{len(pred)}', 
						transform=axes[i].transAxes, 
						bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
			
			axes[i].set_xlabel('Actual Minutes')
			axes[i].set_ylabel('Predicted Minutes')
			axes[i].set_title(name)
			axes[i].grid(True, alpha=0.3)
		else:
			axes[i].text(0.5, 0.5, 'No valid predictions', ha='center', va='center', 
						transform=axes[i].transAxes)
			axes[i].set_title(name)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/prediction_comparison_plots_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Prediction comparison plots saved!', 'green'))

def create_error_distribution_plots(y_test, methods, save_folder):
	"""
	Create error distribution plots for all methods.
	"""
	print(colored('\nCreating error distribution plots...', 'blue'))
	
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	fig.suptitle('Error Distribution - All Methods (Canadian University Basketball)', fontsize=16)
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
	
	for i, (name, pred) in enumerate(methods.items()):
		# Filter out NaN predictions
		valid_mask = ~np.isnan(pred)
		if valid_mask.sum() > 0:
			valid_pred = pred[valid_mask]
			valid_actual = y_test.iloc[valid_mask]
			errors = valid_actual - valid_pred
			
			axes[i].hist(errors, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
			axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.8)
			
			# Add error statistics
			mean_error = errors.mean()
			std_error = errors.std()
			axes[i].text(0.05, 0.95, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}', 
						transform=axes[i].transAxes,
						bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
			
			axes[i].set_xlabel('Prediction Error (Actual - Predicted)')
			axes[i].set_ylabel('Frequency')
			axes[i].set_title(name)
			axes[i].grid(True, alpha=0.3)
		else:
			axes[i].text(0.5, 0.5, 'No valid predictions', ha='center', va='center', 
						transform=axes[i].transAxes)
			axes[i].set_title(name)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/error_distribution_plots_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Error distribution plots saved!', 'green'))

def print_summary(results):
	"""
	Print a summary of the comparison results.
	"""
	print(colored('\n' + '='*60, 'cyan'))
	print(colored('BASELINE COMPARISON SUMMARY', 'cyan'))
	print(colored('='*60, 'cyan'))
	
	df_results = pd.DataFrame(results).T
	
	# Find best model by R²
	best_model = df_results['R²'].idxmax()
	best_r2 = df_results.loc[best_model, 'R²']
	
	print(f"\n🏆 Best Method: {best_model} (R² = {best_r2:.4f})")
	
	print("\n📊 Performance Comparison:")
	print("-" * 50)
	for method, metrics in results.items():
		print(f"{method}:")
		print(f"  R² Score: {metrics['R²']:.4f}")
		print(f"  RMSE: {metrics['RMSE']:.2f} minutes")
		print(f"  MAE: {metrics['MAE']:.2f} minutes")
		print(f"  Valid Predictions: {metrics['Valid_Predictions']}/{metrics['Total_Predictions']}")
		print()
	
	# Calculate improvement over best baseline
	baseline_methods = [m for m in results.keys() if 'Rolling' in m]
	if baseline_methods:
		best_baseline = max(baseline_methods, key=lambda x: results[x]['R²'])
		best_baseline_r2 = results[best_baseline]['R²']
		
		if 'Linear Regression' in results:
			lr_r2 = results['Linear Regression']['R²']
			improvement = ((lr_r2 - best_baseline_r2) / best_baseline_r2) * 100
			
			print(f"📈 Linear Regression vs {best_baseline}:")
			print(f"   Improvement: {improvement:+.1f}%")
			
			if improvement > 0:
				print("   ✅ Linear Regression outperforms the baseline!")
			else:
				print("   ❌ Linear Regression does not outperform the baseline.")

if __name__ == "__main__":
	main(opt['--file_name'], opt['--save_folder']) 