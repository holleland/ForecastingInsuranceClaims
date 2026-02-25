# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:38:52 2024

@author: s15052
"""
# -*- coding: utf-8 -*-
"""
CNN for Bergen no splines
"""
import pandas as pd
import numpy as np
import keras_tuner as kt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.random import set_seed


# data: 
data = pd.read_csv('data/toy_train_bergen.csv',
                   sep = ";",decimal=",")
Y_train = data["claim_cat"] == "many claims"
X_train = data.drop(labels = ["claim_cat", "lead_time", "date", "area", "yday", "set","obs"],
                    axis = 1) 
# Reshape X to add the third dimension (required by Conv1D)
X_train = np.expand_dims(X_train, axis=-1)


# Prepare test data: 
test = pd.read_csv('data/toy_test_bergen.csv',
                   sep = ";",decimal=",")
Y_test = test["claim_cat"] == "many claims"
X_test = test.drop(labels = ["claim_cat", "lead_time", "date", 
                             "area", "yday", "set","obs"], 
                   axis= 1) 
X_test = np.expand_dims(X_test, axis=-1)

# Define the model building function
def build_model(hp):
    seed_value = 422
    random.seed(seed_value)
    np.random.seed(seed_value)
    set_seed(seed_value)
    model = Sequential()
    
    # Convolutional layer 1
    model.add(Conv1D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv1_kernel_size', values=[3, 5, 7]),
        activation='relu',
        input_shape=(51, 1)
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional layer 2
    model.add(Conv1D(
        filters=hp.Int('conv2_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv2_kernel_size', values=[3, 5, 7]),
        activation='relu',
        kernel_regularizer=l1(hp.Float('conv2_l1', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional layer 3 (optional)
    model.add(Conv1D(
        filters=hp.Int('conv3_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv3_kernel_size', values=[3, 5, 7]),
        activation='relu',
        kernel_regularizer=l1(hp.Float('conv3_l1', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Flattening layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(
        units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))

    # Dropout layer (optional)
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.7, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(
                      learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    
    return model

# Set up the tuner
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=20,  # Number of different hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to be built and evaluated for each combination of hyperparameters
    directory='bergen_nosplines_3',
    project_name='cnn_hyperparameter_tuning_time_ex2017'
)

# Display a summary of the search space
tuner.search_space_summary()

# Perform the search
tuner.search(X_train, Y_train, epochs=50, validation_split=0.2, verbose=1)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()

# Retrieve the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hyperparameters.values)

loss, auc = best_model.evaluate(X_test, Y_test)


predictions_prob = best_model.predict(X_test)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions_prob, columns=['Probability'])
predictions_df.to_csv(
    'predictions/toy_CNN_bergen_test.csv', 
    index=False)


insample_predictions_prob = best_model.predict(X_train)
insample_predictions_df = pd.DataFrame(insample_predictions_prob, columns=['Probability'])
insample_predictions_df.to_csv(
    'predictions/toy_CNN_bergen_train.csv', 
    index=False)


# -*- coding: utf-8 -*-
"""
CNN for Oslo no splines
"""
# data: 
data = pd.read_csv('data/toy_train_oslo.csv',
                   sep = ";",decimal=",")
Y_train = data["claim_cat"] == "many claims"
X_train = data.drop(labels = ["claim_cat", "lead_time", "date", "area", "yday", "set","obs"],
                    axis = 1) 
# Reshape X to add the third dimension (required by Conv1D)
X_train = np.expand_dims(X_train, axis=-1)


# Prepare test data: 
test = pd.read_csv('data/toy_test_oslo.csv',
                   sep = ";",decimal=",")
Y_test = test["claim_cat"] == "many claims"
X_test = test.drop(labels = ["claim_cat", "lead_time", "date", "area", "yday", "set","obs"], 
                   axis= 1) 
X_test = np.expand_dims(X_test, axis=-1)

# Set seed to e

# Define the model building function
def build_model(hp):
    seed_value = 422
    random.seed(seed_value)
    np.random.seed(seed_value)
    set_seed(seed_value)
    model = Sequential()
    
    # Convolutional layer 1
    model.add(Conv1D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv1_kernel_size', values=[3, 5, 7]),
        activation='relu',
        input_shape=(51, 1)
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional layer 2
    model.add(Conv1D(
        filters=hp.Int('conv2_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv2_kernel_size', values=[3, 5, 7]),
        activation='relu',
        kernel_regularizer=l1(hp.Float('conv2_l1', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional layer 3 (optional)
    model.add(Conv1D(
        filters=hp.Int('conv3_filters', min_value=16, max_value=128, step=16),
        kernel_size=hp.Choice('conv3_kernel_size', values=[3, 5, 7]),
        activation='relu',
        kernel_regularizer=l1(hp.Float('conv3_l1', min_value=1e-4, max_value=1e-2, sampling='LOG'))
    ))
    model.add(MaxPooling1D(pool_size=2))

    # Flattening layer
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(
        units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))

    # Dropout layer (optional)
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.7, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(
                      learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    
    return model

# Set up the tuner
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=20,  # Number of different hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to be built and evaluated for each combination of hyperparameters
    directory='oslo_nosplines_33',
    project_name='cnn_hyperparameter_tuning_time_ex2017'
)

# Display a summary of the search space
tuner.search_space_summary()

# Perform the search
tuner.search(X_train, Y_train, epochs=50, validation_split=0.2, verbose=1)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()

# Retrieve the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hyperparameters.values)

loss, auc = best_model.evaluate(X_test, Y_test)


predictions_prob = best_model.predict(X_test)

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions_prob, columns=['Probability'])
# Save the DataFrame to a CSV file
#predictions_df.to_csv(
#    'C:/Users/s15052/Dropbox/NHH/Prosjekter/ForecastingInsurancepredictions/CNN_1_oslo_test_without_splines.csv', 
#    index=False)
#predictions_df.to_csv(
#    'C:/Users/s15052/Dropbox/NHH/Prosjekter/ForecastingInsurancepredictions/CNN_2_oslo_test_without_splines.csv', 
#    index=False)
predictions_df.to_csv(
    'predictions/toy_CNN_oslo_test.csv', 
    index=False)


insample_predictions_prob = best_model.predict(X_train)
insample_predictions_df = pd.DataFrame(insample_predictions_prob, columns=['Probability'])
insample_predictions_df.to_csv(
    'predictions/toy_CNN_oslo_train.csv', 
    index=False)
