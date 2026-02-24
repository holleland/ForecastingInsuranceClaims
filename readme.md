Short-Term Management of Water-Damage Claim Risk Using Ensemble
Precipitation Forecasts
================

This repository contains necessary code for reproducing results
presented in the paper **Short-Term Management of Water-Damage Claim
Risk Using Ensemble Precipitation Forecasts** by Håkon Otneim, Etienne
Dunn-Sigouin, Sondre Hølleland, Mahsa Gorji and Geir Drage Berentsen.
Since the insurance data is not publicly available, we have generated
data from one of the models using the script *R/0_making_toydata.R*.
This uses the estimated logistic regression model for Oslo and Bergen to
generate a toy dataset. Thus, you will not be able to obtain the
specific results, but you have a running example.

The script *0_run_scripts.R* explains the order of how things should be
run. The resulting predictions are stored the predictions folder and
figures are stored in the figures folder. The script is included below.

``` r
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
```
