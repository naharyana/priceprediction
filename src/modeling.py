import util as utils

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

def create_model_param():
    """Create the model objects"""    
    xgb_params = {
        'n_estimators': [5, 10, 25, 50]
    }

    # Create model params
    list_of_param = {
        'XGBRegressor': xgb_params
    }

    return list_of_param

def create_model_object():
    """Create the model objects"""
    print("Creating model objects")

    # Create model objects
    lr = LinearRegression()
    xgb = XGBRegressor()

    # Create list of model
    list_of_model = [
        {'model_name': lr.__class__.__name__, 'model_object': lr},
        {'model_name': xgb.__class__.__name__, 'model_object': xgb}
    ]

    return list_of_model

def train_model(config_data, return_file=True):
    """Function to get the best model"""
    # Load dataset
    X_train = utils.pickle_load(config_data['train_clean_path'][0])
    y_train = utils.pickle_load(config_data['train_clean_path'][1])
    X_valid = utils.pickle_load(config_data['valid_clean_path'][0])
    y_valid = utils.pickle_load(config_data['valid_clean_path'][1])

    # Create list of models
    list_of_model = create_model_object()

    # List of trained model
    list_of_tuned_model = {}

    # Train model
    for base_model in list_of_model:
        # Current condition
        model_name = base_model['model_name']
        model_obj = copy.deepcopy(base_model['model_object'])

        # Debug message
        print('Training model :', model_name)
        
        # Train model
        model_obj.fit(X_train, y_train)

        # Predict
        y_pred_train = model_obj.predict(X_train)
        y_pred_valid = model_obj.predict(X_valid)
        
        # Get score
        train_score = mean_absolute_error(y_train, y_pred_train)
        valid_score = mean_absolute_error(y_valid, y_pred_valid)

        # Append
        list_of_tuned_model[model_name] = {
            'model': model_obj,
            'train_mae': train_score,
            'valid_mae': valid_score
        }

        print("Done training")
        print("")

    # Dump data
    utils.pickle_dump(list_of_model, config_data['list_of_model_path'])
    utils.pickle_dump(list_of_tuned_model, config_data['list_of_tuned_model_path'])

    if return_file:
        return list_of_model, list_of_tuned_model    

def get_best_model(config_data, return_file=True):
    """Function to get the best model"""
    # Load tuned model
    list_of_tuned_model = utils.pickle_load(config_data['list_of_tuned_model_path'])

    # Get the best model
    best_model_name = None
    best_model = None
    best_performance = 99999
    best_model_param = None

    for model_name, model in list_of_tuned_model.items():
        if model['valid_mae'] < best_performance:
            best_model_name = model_name
            best_model = model['model']
            best_performance = model['valid_mae']

    # Dump the best model
    utils.pickle_dump(best_model, config_data['best_model_path'])

    # Print
    print('=============================================')
    print('Best model        :', best_model_name)
    print('Metric score      :', best_performance)
    print('=============================================')

    if return_file:
        return best_model

if __name__ == '__main__':
    # 1. Load configuration file
    config_data = utils.load_config()

    # 2. Train & Optimize the model
    train_model(config_data)

    # 3. Get the best model
    get_best_model(config_data)