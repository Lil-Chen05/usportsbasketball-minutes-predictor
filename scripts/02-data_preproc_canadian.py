# author: Adapted for Canadian University Basketball Data
# date: 2024-12-19

"""
This script takes raw data from the 'data' folder in the project repository to perform data preprocessing work.
Data preprocessing mainly includes making rolling and ewm features used for predicting 'Mins'.

Both the input file path+name and the save folder are required as inputs.

Usage: 02-data_preproc_canadian.py --input_path_file=<file_name> --save_folder=<save_folder>

Options:
--input_path_file=<file_name>         path and file name of the input data to be pre-processed
--save_folder=<save_folder>	    folder to save the processed output data

Example: python scripts/02-data_preproc_canadian.py --input_path_file=data/2022-24_playerBoxScore.csv --save_folder=data
"""

# Loading the required packages
# Data proc
import pandas as pd
pd.set_option('mode.chained_assignment', None) # turn off warning message of SettingWithCopyWarning 
import numpy as np
# Other Packages
from docopt import docopt
from tqdm import tqdm
import sys
import os
from termcolor import colored
# Ignore warnings from packages in models
import warnings
warnings.simplefilter("ignore")

opt = docopt(__doc__)

def main(input_path_file, save_folder):
	# Load the original data from csv
	# e.g. '2022-24_playerBoxScore.csv'
	
	print(colored("\nWARNING: This script takes about 1 minute to run\n", 'yellow'))

	# Validate the file-path to load file
	path_str = str(input_path_file)
	if os.path.exists(path_str) == False:
		print(colored('ERROR: Path to file is not valid!', 'red'))
	try:
		df = pd.read_csv(path_str)
		print(colored('Data loaded successfully!', 'green'))
	except:
		print(colored("ERROR: Data can't be loaded!", 'red'))
		raise

	# Validate the save_foler directory exists or make folder
	if os.path.exists(str(save_folder)) == False:
		try:
			os.makedirs(save_folder)
		except:
			print(colored('ERROR: Path to save directory is not valid!', 'red'))
			raise
	
	#######################################
	#### Data preprocessing starts here####
	#######################################

	df = df.dropna()

	# Convert Date to datetime
	df['Date'] = pd.to_datetime(df['Date'])

	# Create player rating feature (similar to NBA but adapted for available stats)
	df['PlayerRating'] = (df['Pts'] + 
					(df['BLK'] * 2) +
					(df['TO'] * -0.5) +
					(df['STL'] * 2) +
					(df['AST'] * 1.5) +
					(df['Reb_T'] * 1.25))

	# Create efficiency metrics
	df['UsageRate'] = (df['FGA'] + 0.44 * df['FTA'] + df['TO']) / df['Mins'].replace(0, 1) * 100
	df['TrueShootingPct'] = df['TS_Pct']
	df['EffectiveFGPct'] = df['eFG_Pct']

	# Create per-minute stats
	df['PtsPerMin'] = df['Pts'] / df['Mins'].replace(0, 1)
	df['AstPerMin'] = df['AST'] / df['Mins'].replace(0, 1)
	df['RebPerMin'] = df['Reb_T'] / df['Mins'].replace(0, 1)

	# test 'PlayerRating' column
	assert df['PlayerRating'].isna().sum() == 0, colored("ERROR: NaN value detected in 'PlayerRating' column!!", 'red')

	# filter columns for modeling
	cols_to_use = ['Date', 'Team', 'Opponent', 'PlayerName', 'Mins', 'PlayerRating', 'StarterFlag', 
				   'UsageRate', 'TrueShootingPct', 'EffectiveFGPct', 'PtsPerMin', 'AstPerMin', 'RebPerMin',
				   'FG_Pct', '3PT_Pct', 'FT_Pct', 'Reb_O', 'Reb_D', 'PF', 'AST', 'TO', 'BLK', 'STL', 'Pts']
	df = df[cols_to_use].copy()

	# Convert StarterFlag to numeric for rolling operations
	df['StarterFlag'] = df['StarterFlag'].astype(int)

	# test on the categorical value replacement
	assert set(df['StarterFlag'].unique()) == {0,1}, colored("ERROR: Categorical value replacement failed!", 'red')

	# make input variables for making rolling features and ewm features
	cols_keep = ['PlayerName', 'Date', 'Team', 'Mins', 'StarterFlag']
	cols_roll = ['Mins','PlayerRating', 'UsageRate', 'TrueShootingPct', 'EffectiveFGPct', 'PtsPerMin', 'AstPerMin', 'RebPerMin']
	windows = [3, 5, 10]  # Smaller windows due to less data
	ewm_alpha = [0.1, 0.2, 0.3, 0.5]  # Fewer alphas due to less data
	agg_funcs = ['mean', 'median']  # Added mean for more features

	df_org = df.copy()
	df = pd.DataFrame() 

	# iterate through names to make new df with rolling and ewm features
	name_list = list(df_org['PlayerName'].unique())

	print(colored("\nData processing in progress:", 'green'))
	for name in tqdm(name_list):
		thisguy = df_org.query("PlayerName == @name").sort_values('Date', ascending=True)
		if len(thisguy) < 3: # Reduced minimum games due to smaller dataset
			continue
		cols_created = []
		
		# make rolling features
		cols_created_rolling, thisguy = make_rolling_features(cols_roll, windows, agg_funcs, thisguy)
		cols_created.extend(cols_created_rolling)
		# test on making rolling features
		assert len(cols_created_rolling) == len(cols_roll) * len(windows) * len(agg_funcs), "Number of created rolling features is wrong!"
		assert thisguy.isna().sum().sum() == 0, "NaN value detected when making rolling features!"

		# make ewm features
		cols_created_ewm, thisguy = make_ewm_features(cols_roll, ewm_alpha, thisguy)
		cols_created.extend(cols_created_ewm)
		# test on making rolling features
		assert len(cols_created_ewm) == len(cols_roll) * len(ewm_alpha) + len(ewm_alpha), "Number of created ewm features is wrong!"
		assert thisguy.isna().sum().sum() == len(ewm_alpha), "Number of ewm features containing NaN should be the same as the length of ewm_alpha!" #ewm_std features should have 1 NaN value for the first row

		# shift created features by 1 row so that it means the "last n games"          
		thisguy_result = meaningful_shift(cols_created, cols_keep, thisguy)

		# append this guy's result table into df
		df = pd.concat((df, thisguy_result), axis=0, ignore_index=True).copy()

		
	df = df.dropna().copy()

	# wrangling part ends, save the result dataframe
	df.to_csv(str(save_folder)+'/player_data_ready_canadian.csv', index=False)
	print(colored('Data successfully saved!', 'green'))

	print(colored('\nData preprocessing complete!', 'green'))
##################################
######## Define Functions ########
##################################
def catg_num_replace(rep_dict, df_input):
	"""
	Replace categorical values with numbers to apply `rolling` to them.

	Parameters:
	-----------
	rep_dict -- (dict) the dictionary used for replacement. format: {'column_name':{categorical_level: number_to_replace}}
	df_input -- (pd DataFrame) the input dataframe which contains the categorical features to be replaced.
	
	Return:
	-----------
	df_output -- (pd DataFrame) the output dataframe
	"""
	df_output = df_input.copy()
	for x in rep_dict.keys():
		df_output[x] = df_input[x].apply(lambda y: rep_dict[x][y])

	return df_output


def make_rolling_features(cols_roll, windows, agg_funcs, df_input):
	"""
	Make and add rolling features to the input dataframe given the columns, windows and aggregate function. 
	And record features created.

	Parameters:
	-----------
	cols_roll -- (list) a list of column names used to make rolling features
	windows -- (list) a list of int used as the window of making rolling features
	agg_funcs -- (list) a list of function names used as aggregate function for making rolling features
	df_input -- (pd DataFrame) the input dataframe which contains the features used for making rolling features
	
	Return:
	-----------
	cols_created_this_roll -- (list) a list of column names of the created rolling features
	df_output -- (pd DataFrame) the output dataframe containing all the rolling features
	"""
	cols_created_this_roll = []
	df_output = df_input.copy()
	for col in cols_roll:
		for t in windows:
			for fun in agg_funcs:
				new_col = col+'_last'+str(t)+'_'+fun
				cols_created_this_roll.append(new_col)
				df_output.loc[:, new_col] = getattr(df_input[col].rolling(t, min_periods=1), fun)().copy()

	return cols_created_this_roll, df_output


def make_ewm_features(cols_roll, ewm_alpha, df_input):
	"""
	Make mean ewm features based on cols_roll, ewm_alpha.
	Make std ewm features for 'Mins'. 
	Return list of column names containing all the ewm features created.

	Parameters:
	-----------
	cols_roll -- (list) a list of column names used to make ewm features
	ewm_alpha -- (list) a list of float used as the alpha parameter for making ewm features
	df_input -- (pd DataFrame) the input dataframe which contains the features used for making ewm features
	
	Return:
	-----------
	cols_created_this_ewm -- (list) a list of column names of the created ewm features
	df_output -- (pd DataFrame) the output dataframe containing all the ewm features
	"""
	cols_created_this_ewm = []
	df_output = df_input.copy()
	for col in cols_roll:
		for alpha in ewm_alpha:
			new_col = col+'_ewm_'+str(alpha)
			cols_created_this_ewm.append(new_col)
			df_output.loc[:, new_col] = df_input[col].ewm(alpha=alpha).mean().copy()
	
	# make std ewm features for 'Mins' only
	for alpha in ewm_alpha:
		new_col = 'Mins_ewm_std_'+str(alpha)
		cols_created_this_ewm.append(new_col)
		df_output.loc[:, new_col] = df_input['Mins'].ewm(alpha=alpha).std().copy()

	return cols_created_this_ewm, df_output


def meaningful_shift(cols_created, cols_keep, df_input):
	"""
	Shift created features by 1 row so that it means the "last n games" and keep the original columns.

	Parameters:
	-----------
	cols_created -- (list) a list of column names of the created features to be shifted
	cols_keep -- (list) a list of column names to be kept without shifting
	df_input -- (pd DataFrame) the input dataframe which contains the features to be shifted
	
	Return:
	-----------
	df_output -- (pd DataFrame) the output dataframe containing shifted features and kept columns
	"""
	df_output = df_input.copy()
	df_output[cols_created] = df_input[cols_created].shift(1).copy()
	df_output = df_output[cols_keep + cols_created].copy()

	return df_output


if __name__ == "__main__":
	main(opt['--input_path_file'], opt['--save_folder']) 