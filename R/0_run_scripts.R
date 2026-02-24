# Run scripts
# Set up directories:
dir.create("predictions")
dir.create("figures")


# Generate toy data:
source("R/1_making_toydata.R")

# Fit models:
# -- Warning!!! require some time ---
source("R/2_models_fit.R")


# Run the CNN model in Python
#python/run_CNN.py

# Then add the prediction from the CNN
source("R/2b_add_CNN_predictions.R")

# Genenerate results figures: 
source("R/3_PRAUC_plots.R") # Takes a few minutes due to bootstrap
source("R/4_reliability.R")





