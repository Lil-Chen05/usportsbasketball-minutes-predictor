# Makefile for Canadian University Basketball Minutes Predictor
# Author: Adapted from NBA Minutes Predictor
# Date: 2024-12-19

# Default target - Canadian pipeline
all: data/player_data_ready_canadian.csv results/EDA-feat_corr_canadian.png results/modelling-score_table_canadian.csv results/model_performance_comparison_canadian.png results/baseline_comparison_canadian.png

# Canadian pipeline (with LightGBM for comparison)
canadian: data/player_data_ready_canadian.csv results/EDA-feat_corr_canadian.png results/modelling-score_table_canadian.csv results/model_performance_comparison_canadian.png results/baseline_comparison_canadian.png

# Canadian pipeline (fixed - no LightGBM)
canadian_fixed: data/player_data_ready_canadian.csv results/EDA-feat_corr_canadian.png results/modelling-score_table_canadian_fixed.csv results/model_performance_comparison_canadian_fixed.png

# Data preprocessing
data/player_data_ready_canadian.csv: data/2022-24_playerBoxScore.csv
	python scripts/02-data_preproc_canadian.py --input_path_file=data/2022-24_playerBoxScore.csv --save_folder=data

# EDA
results/EDA-feat_corr_canadian.png: data/player_data_ready_canadian.csv
	python scripts/03-EDA_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results

# Model fitting
results/modelling-score_table_canadian.csv: data/player_data_ready_canadian.csv
	python scripts/04-model_fit_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results

results/modelling-score_table_canadian_fixed.csv: data/player_data_ready_canadian.csv
	python scripts/04-model_fit_canadian_fixed.py --file_name=player_data_ready_canadian.csv --save_folder=results

# Model visualization
results/model_performance_comparison_canadian.png: data/player_data_ready_canadian.csv
	python scripts/05-model_visualization_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results

results/model_performance_comparison_canadian_fixed.png: data/player_data_ready_canadian.csv
	python scripts/05-model_visualization_canadian_fixed.py --file_name=player_data_ready_canadian.csv --save_folder=results

# Baseline comparison
results/baseline_comparison_canadian.png: data/player_data_ready_canadian.csv
	python scripts/06-baseline_comparison_canadian.py --file_name=player_data_ready_canadian.csv --save_folder=results

# Generate comprehensive report
report: all
	python scripts/07-generate_report.py --save_folder=results

# Clean up
clean:
	rm -f data/player_data_ready_canadian.csv
	rm -f results/EDA-feat_corr_canadian.png
	rm -f results/EDA-feat_corr_canadian.png
	rm -f results/EDA-correl_df_neg_canadian.csv
	rm -f results/EDA-correl_df_pos_canadian.csv
	rm -f results/EDA-feature_importance_canadian.png
	rm -f results/EDA-feature_importance_canadian.csv
	rm -f results/EDA-hist_y_canadian.png
	rm -f results/modelling-score_table_canadian.csv
	rm -f results/modelling-score_table_canadian_fixed.csv
	rm -f results/modelling-gbm_importance_canadian.png
	rm -f results/modelling-rf_importance_canadian_fixed.png
	rm -f results/model_performance_comparison_canadian.png
	rm -f results/model_performance_comparison_canadian_fixed.png
	rm -f results/model_performance_detailed_canadian.csv
	rm -f results/model_performance_detailed_canadian_fixed.csv
	rm -f results/baseline_comparison_canadian.png
	rm -f results/baseline_comparison_detailed_canadian.csv
	rm -f results/prediction_comparison_plots_canadian.png
	rm -f results/error_distribution_plots_canadian.png
	rm -f results/prediction_scatter_plots_canadian.png
	rm -f results/prediction_scatter_plots_canadian_fixed.png
	rm -f results/residual_plots_canadian.png
	rm -f results/residual_plots_canadian_fixed.png
	rm -f results/feature_importance_comparison_canadian.png
	rm -f results/feature_importance_comparison_canadian_fixed.png
	rm -f results/canadian_basketball_report.pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Run the complete Canadian pipeline (default)"
	@echo "  canadian         - Run the Canadian pipeline (with LightGBM for comparison)"
	@echo "  canadian_fixed   - Run the Canadian pipeline (no LightGBM, Linear Regression + Random Forest only)"
	@echo "  report           - Generate comprehensive PDF report with all results"
	@echo "  clean            - Remove all generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Note: data/2022-24_playerBoxScore.csv must exist for the pipeline to work"

.PHONY: all canadian canadian_fixed report clean help
