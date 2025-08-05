# author: Adapted for Canadian University Basketball Data
# date: 2024-12-19

"""
This script takes processed data from the 'data' folder in the project repository
and creates exploratory data analysis (EDA) plots and tables.
The EDA includes correlation analysis, distribution plots, and feature importance analysis.

Both the file name and the save folder are required as inputs.

Usage: 03-EDA_canadian.py --file_name=<file_name> --save_folder=<save_folder>

Options:
--file_name=<file_name>         File name of the processed features and targets
--save_folder=<save_folder>	Folder to save all figures and csv files produced

Example: python scripts/03-EDA_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results
"""

# Loading the required packages
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Numerical Packages
import numpy as np
import pandas as pd
# SKLearn Packages
from sklearn.feature_selection import mutual_info_regression
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
	# e.g. 'player_data_ready_canadian.csv'

	print(colored("\nWARNING: This script takes about 30 seconds to run\n", 'yellow'))

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

	# Create EDA plots and tables
	correlation_analysis(X, y, save_folder)
	distribution_analysis(y, save_folder)
	feature_importance_analysis(X, y, save_folder)

	print(colored('\nEDA complete!', 'green'))

def preprocess(data):
	"""
	Preprocess the data for EDA analysis.

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

	return X, y

def correlation_analysis(X, y, save_folder):
	"""
	Perform correlation analysis between features and target.

	Parameters:
	-----------
	X -- (pd DataFrame) the feature matrix
	y -- (pd Series) the target variable
	save_folder -- (str) folder to save results
	"""
	print(colored('\nPerforming correlation analysis...', 'blue'))

	# Add target to correlation analysis
	X_with_target = X.copy()
	X_with_target['Mins'] = y

	# Calculate correlation matrix
	corr_matrix = X_with_target.corr()

	# Get correlations with target
	target_corr = corr_matrix['Mins'].sort_values(ascending=False)
	
	# Save top positive and negative correlations
	top_pos = target_corr[target_corr > 0].head(20)
	top_neg = target_corr[target_corr < 0].head(20)

	# Save to CSV
	top_pos.to_csv(f'{save_folder}/EDA-correl_df_pos_canadian.csv')
	top_neg.to_csv(f'{save_folder}/EDA-correl_df_neg_canadian.csv')

	# Create correlation heatmap for top features
	top_features = list(top_pos.index[1:11]) + list(top_neg.index[:10])  # Exclude 'Mins' from positive
	top_corr = corr_matrix.loc[top_features, top_features]

	plt.figure(figsize=(12, 10))
	sns.heatmap(top_corr, annot=True, cmap='RdBu_r', center=0, square=True)
	plt.title('Correlation Heatmap - Top Features (Canadian University Basketball)')
	plt.tight_layout()
	plt.savefig(f'{save_folder}/EDA-feat_corr_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()

	print(colored('Correlation analysis complete!', 'green'))

def distribution_analysis(y, save_folder):
	"""
	Analyze the distribution of the target variable.

	Parameters:
	-----------
	y -- (pd Series) the target variable
	save_folder -- (str) folder to save results
	"""
	print(colored('\nPerforming distribution analysis...', 'blue'))

	plt.figure(figsize=(10, 6))
	plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
	plt.xlabel('Minutes Played')
	plt.ylabel('Frequency')
	plt.title('Distribution of Minutes Played (Canadian University Basketball)')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(f'{save_folder}/EDA-hist_y_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()

	print(colored('Distribution analysis complete!', 'green'))

def feature_importance_analysis(X, y, save_folder):
	"""
	Perform feature importance analysis using mutual information.

	Parameters:
	-----------
	X -- (pd DataFrame) the feature matrix
	y -- (pd Series) the target variable
	save_folder -- (str) folder to save results
	"""
	print(colored('\nPerforming feature importance analysis...', 'blue'))

	# Calculate mutual information scores
	mi_scores = mutual_info_regression(X, y, random_state=42)
	mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
	mi_scores_df = mi_scores_df.sort_values('MI_Score', ascending=False)

	# Plot top 20 features
	plt.figure(figsize=(12, 8))
	top_features = mi_scores_df.head(20)
	plt.barh(range(len(top_features)), top_features['MI_Score'])
	plt.yticks(range(len(top_features)), top_features['Feature'])
	plt.xlabel('Mutual Information Score')
	plt.title('Feature Importance - Mutual Information (Canadian University Basketball)')
	plt.gca().invert_yaxis()
	plt.tight_layout()
	plt.savefig(f'{save_folder}/EDA-feature_importance_canadian.png', dpi=300, bbox_inches='tight')
	plt.close()

	# Save feature importance scores
	mi_scores_df.to_csv(f'{save_folder}/EDA-feature_importance_canadian.csv', index=False)

	print(colored('Feature importance analysis complete!', 'green'))

if __name__ == "__main__":
	main(opt['--file_name'], opt['--save_folder']) 