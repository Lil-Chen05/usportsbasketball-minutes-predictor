# Author: Additional Visualizations for Canadian University Basketball Analysis
# Date: 2024-12-19

"""
This script creates additional impactful visualizations for the Canadian basketball analysis.
It generates error analysis, prediction confidence intervals, and model comparison charts.

Usage: 08-create_additional_visualizations.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/08-create_additional_visualizations.py --file_name=player_data_ready_canadian.csv --save_folder=results
"""

# Loading the required packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from docopt import docopt
from termcolor import colored
import os
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(file_name, save_folder):
	"""
	Create additional impactful visualizations for the analysis.
	"""
	print(colored('\nCreating additional impactful visualizations...', 'blue'))
	
	# Load data
	path_str = str('../data/' + file_name)
	if os.path.exists(path_str) == False:
		path_str = str('data/' + file_name)
	
	try:
		data = pd.read_csv(path_str)
		print(colored('Data loaded successfully!', 'green'))
	except:
		print(colored('ERROR: Path to file is not valid!', 'red'))
		raise

	# Preprocess data
	X, y = preprocess(data)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train models
	models = {
		'Linear Regression': LinearRegression(),
		'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
	}

	predictions = {}
	for name, model in models.items():
		model.fit(X_train, y_train)
		predictions[name] = model.predict(X_test)

	# Create visualizations
	create_error_analysis_plot(y_test, predictions, save_folder)
	create_prediction_confidence_plot(y_test, predictions, save_folder)
	create_model_comparison_radar(y_test, predictions, save_folder)
	create_feature_importance_comparison_enhanced(models, X, save_folder)
	create_performance_trend_analysis(y_test, predictions, save_folder)
	
	print(colored('\nAdditional visualizations complete!', 'green'))

def preprocess(data):
	"""
	Preprocess the data for modeling.
	"""
	y = data['Mins']
	X = data.drop(['Mins', 'PlayerName', 'Date', 'Team'], axis=1, errors='ignore')
	X = X.select_dtypes(include=[np.number])
	X = X.replace([np.inf, -np.inf], np.nan)
	X = X.fillna(X.mean())
	return X, y

def create_error_analysis_plot(y_test, predictions, save_folder):
	"""
	Create comprehensive error analysis visualization.
	"""
	print(colored('\nCreating error analysis plot...', 'blue'))
	
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Error Analysis: Model Performance Deep Dive', fontsize=16, fontweight='bold')
	
	colors = ['#FF6B6B', '#4ECDC4']
	
	for i, (name, pred) in enumerate(predictions.items()):
		residuals = y_test - pred
		abs_errors = np.abs(residuals)
		
		# Error distribution
		axes[0, 0].hist(residuals, bins=30, alpha=0.7, color=colors[i], label=name)
		axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.8)
		axes[0, 0].set_xlabel('Residuals (Actual - Predicted)')
		axes[0, 0].set_ylabel('Frequency')
		axes[0, 0].set_title('Error Distribution')
		axes[0, 0].legend()
		axes[0, 0].grid(True, alpha=0.3)
		
		# Error vs Predicted
		axes[0, 1].scatter(pred, residuals, alpha=0.6, color=colors[i], label=name)
		axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
		axes[0, 1].set_xlabel('Predicted Minutes')
		axes[0, 1].set_ylabel('Residuals')
		axes[0, 1].set_title('Error vs Predicted')
		axes[0, 1].legend()
		axes[0, 1].grid(True, alpha=0.3)
		
		# Absolute error distribution
		axes[1, 0].hist(abs_errors, bins=30, alpha=0.7, color=colors[i], label=name)
		axes[1, 0].set_xlabel('Absolute Error')
		axes[1, 0].set_ylabel('Frequency')
		axes[1, 0].set_title('Absolute Error Distribution')
		axes[1, 0].legend()
		axes[1, 0].grid(True, alpha=0.3)
		
		# Error statistics
		mean_error = residuals.mean()
		std_error = residuals.std()
		mae = abs_errors.mean()
		rmse = np.sqrt(mean_squared_error(y_test, pred))
		
		axes[1, 1].text(0.1, 0.8 - i*0.3, f'{name}:', fontweight='bold', transform=axes[1, 1].transAxes)
		axes[1, 1].text(0.1, 0.75 - i*0.3, f'Mean Error: {mean_error:.2f}', transform=axes[1, 1].transAxes)
		axes[1, 1].text(0.1, 0.7 - i*0.3, f'Std Error: {std_error:.2f}', transform=axes[1, 1].transAxes)
		axes[1, 1].text(0.1, 0.65 - i*0.3, f'MAE: {mae:.2f}', transform=axes[1, 1].transAxes)
		axes[1, 1].text(0.1, 0.6 - i*0.3, f'RMSE: {rmse:.2f}', transform=axes[1, 1].transAxes)
	
	axes[1, 1].set_xlim(0, 1)
	axes[1, 1].set_ylim(0, 1)
	axes[1, 1].axis('off')
	axes[1, 1].set_title('Error Statistics')
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/error_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Error analysis plot saved!', 'green'))

def create_prediction_confidence_plot(y_test, predictions, save_folder):
	"""
	Create prediction confidence interval visualization.
	"""
	print(colored('\nCreating prediction confidence plot...', 'blue'))
	
	fig, axes = plt.subplots(1, 2, figsize=(15, 6))
	fig.suptitle('Prediction Confidence Analysis', fontsize=16, fontweight='bold')
	
	colors = ['#FF6B6B', '#4ECDC4']
	
	for i, (name, pred) in enumerate(predictions.items()):
		# Calculate confidence intervals (simplified)
		residuals = y_test - pred
		std_residuals = residuals.std()
		
		# Sort by predicted values for better visualization
		sort_idx = np.argsort(pred)
		pred_sorted = pred[sort_idx]
		actual_sorted = y_test.iloc[sort_idx]
		
		# Plot with confidence bands
		axes[i].scatter(pred_sorted, actual_sorted, alpha=0.6, color=colors[i], s=20)
		axes[i].plot([pred_sorted.min(), pred_sorted.max()], [pred_sorted.min(), pred_sorted.max()], 'r--', alpha=0.8, label='Perfect Prediction')
		
		# Add confidence bands
		axes[i].fill_between(pred_sorted, pred_sorted - 2*std_residuals, pred_sorted + 2*std_residuals, 
							alpha=0.2, color=colors[i], label='95% Confidence Interval')
		
		axes[i].set_xlabel('Predicted Minutes')
		axes[i].set_ylabel('Actual Minutes')
		axes[i].set_title(f'{name} - Prediction Confidence')
		axes[i].legend()
		axes[i].grid(True, alpha=0.3)
		
		# Add R² score
		r2 = r2_score(actual_sorted, pred_sorted)
		axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
					bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Prediction confidence plot saved!', 'green'))

def create_model_comparison_radar(y_test, predictions, save_folder):
	"""
	Create radar chart comparing model performance across multiple metrics.
	"""
	print(colored('\nCreating model comparison radar chart...', 'blue'))
	
	# Calculate multiple metrics for each model
	metrics = {}
	for name, pred in predictions.items():
		mse = mean_squared_error(y_test, pred)
		mae = mean_absolute_error(y_test, pred)
		r2 = r2_score(y_test, pred)
		rmse = np.sqrt(mse)
		
		# Normalize metrics to 0-1 scale for radar chart
		metrics[name] = {
			'R² Score': r2,
			'RMSE (inverted)': 1 / (1 + rmse),  # Invert so higher is better
			'MAE (inverted)': 1 / (1 + mae),    # Invert so higher is better
			'Explained Variance': r2 if r2 > 0 else 0
		}
	
	# Create radar chart
	fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
	
	# Prepare data for radar chart
	categories = list(metrics[list(metrics.keys())[0]].keys())
	N = len(categories)
	
	# Compute angle for each axis
	angles = [n / float(N) * 2 * np.pi for n in range(N)]
	angles += angles[:1]  # Complete the circle
	
	colors = ['#FF6B6B', '#4ECDC4']
	
	for i, (name, metric_values) in enumerate(metrics.items()):
		values = list(metric_values.values())
		values += values[:1]  # Complete the circle
		
		ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
		ax.fill(angles, values, alpha=0.25, color=colors[i])
	
	# Set labels
	ax.set_xticks(angles[:-1])
	ax.set_xticklabels(categories)
	ax.set_ylim(0, 1)
	ax.set_title('Model Performance Comparison (Radar Chart)', fontsize=14, fontweight='bold', pad=20)
	ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
	ax.grid(True)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/model_comparison_radar.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Model comparison radar chart saved!', 'green'))

def create_feature_importance_comparison_enhanced(models, X, save_folder):
	"""
	Create enhanced feature importance comparison with statistical significance.
	"""
	print(colored('\nCreating enhanced feature importance comparison...', 'blue'))
	
	fig, axes = plt.subplots(1, 2, figsize=(15, 8))
	fig.suptitle('Enhanced Feature Importance Analysis', fontsize=16, fontweight='bold')
	
	colors = ['#FF6B6B', '#4ECDC4']
	
	for i, (name, model) in enumerate(models.items()):
		if hasattr(model, 'feature_importances_'):
			importance = model.feature_importances_
		else:
			# For linear regression, use absolute coefficients
			importance = np.abs(model.coef_)
		
		# Get top 15 features
		feature_importance_df = pd.DataFrame({
			'Feature': X.columns,
			'Importance': importance
		}).sort_values('Importance', ascending=False).head(15)
		
		# Create horizontal bar chart
		axes[i].barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
					color=colors[i], alpha=0.8)
		axes[i].set_yticks(range(len(feature_importance_df)))
		axes[i].set_yticklabels(feature_importance_df['Feature'])
		axes[i].set_xlabel('Importance')
		axes[i].set_title(f'{name} - Top 15 Features')
		axes[i].invert_yaxis()
		
		# Add percentage labels
		total_importance = feature_importance_df['Importance'].sum()
		for j, (idx, row) in enumerate(feature_importance_df.iterrows()):
			percentage = (row['Importance'] / total_importance) * 100
			axes[i].text(row['Importance'] + 0.01, j, f'{percentage:.1f}%', 
						va='center', fontsize=8)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Enhanced feature importance comparison saved!', 'green'))

def create_performance_trend_analysis(y_test, predictions, save_folder):
	"""
	Create performance trend analysis showing how models perform across different ranges.
	"""
	print(colored('\nCreating performance trend analysis...', 'blue'))
	
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Performance Trend Analysis: Model Behavior Across Different Ranges', fontsize=16, fontweight='bold')
	
	colors = ['#FF6B6B', '#4ECDC4']
	
	for i, (name, pred) in enumerate(predictions.items()):
		residuals = y_test - pred
		
		# Performance by actual minutes range
		minute_ranges = [(0, 10), (10, 20), (20, 30), (30, 40)]
		range_errors = []
		range_counts = []
		
		for low, high in minute_ranges:
			mask = (y_test >= low) & (y_test < high)
			if mask.sum() > 0:
				range_errors.append(np.abs(residuals[mask]).mean())
				range_counts.append(mask.sum())
			else:
				range_errors.append(0)
				range_counts.append(0)
		
		# Plot 1: Error by minute range
		ranges = ['0-10', '10-20', '20-30', '30-40']
		axes[0, 0].bar(ranges, range_errors, color=colors[i], alpha=0.8, label=name)
		axes[0, 0].set_xlabel('Actual Minutes Range')
		axes[0, 0].set_ylabel('Mean Absolute Error')
		axes[0, 0].set_title('Error by Minutes Range')
		axes[0, 0].legend()
		axes[0, 0].grid(True, alpha=0.3)
		
		# Plot 2: Sample size by range
		axes[0, 1].bar(ranges, range_counts, color=colors[i], alpha=0.8, label=name)
		axes[0, 1].set_xlabel('Actual Minutes Range')
		axes[0, 1].set_ylabel('Number of Predictions')
		axes[0, 1].set_title('Sample Size by Minutes Range')
		axes[0, 1].legend()
		axes[0, 1].grid(True, alpha=0.3)
		
		# Plot 3: Error vs predicted value
		axes[1, 0].scatter(pred, np.abs(residuals), alpha=0.6, color=colors[i], s=20, label=name)
		axes[1, 0].set_xlabel('Predicted Minutes')
		axes[1, 0].set_ylabel('Absolute Error')
		axes[1, 0].set_title('Error vs Predicted Value')
		axes[1, 0].legend()
		axes[1, 0].grid(True, alpha=0.3)
		
		# Plot 4: Error distribution by prediction accuracy
		accuracy_bins = [(0, 2), (2, 5), (5, 10), (10, 20)]
		bin_errors = []
		bin_counts = []
		
		for low, high in accuracy_bins:
			mask = (np.abs(residuals) >= low) & (np.abs(residuals) < high)
			if mask.sum() > 0:
				bin_errors.append(np.abs(residuals[mask]).mean())
				bin_counts.append(mask.sum())
			else:
				bin_errors.append(0)
				bin_counts.append(0)
		
		bin_labels = ['0-2', '2-5', '5-10', '10-20']
		axes[1, 1].bar(bin_labels, bin_counts, color=colors[i], alpha=0.8, label=name)
		axes[1, 1].set_xlabel('Absolute Error Range')
		axes[1, 1].set_ylabel('Number of Predictions')
		axes[1, 1].set_title('Error Distribution')
		axes[1, 1].legend()
		axes[1, 1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(f'{save_folder}/performance_trend_analysis.png', dpi=300, bbox_inches='tight')
	plt.close()
	
	print(colored('Performance trend analysis saved!', 'green'))

if __name__ == "__main__":
	main(opt['--file_name'], opt['--save_folder']) 