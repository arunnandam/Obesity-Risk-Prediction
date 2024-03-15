# All the model training code

# Importing the packages

import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from dataclasses import dataclass
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model, load_object
all_parameters = False #Set it to True if you want to run all hyperparameters.
# Also uncomment models

@dataclass
class ModelTrainerConfig:
    '''
    Defining the model pickle path
    '''
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, mapping, preprocessor_path):
        try:
            logging.info("Splitting Train and Test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("hijiij")
            logging.info(X_test)
            logging.info(y_test)

            # All the models we are using are stored in utils
            models = {
                #"DecisonTree": DecisionTreeClassifier(),
                #"RandomForest": RandomForestClassifier(),
                "XGBoost": xgb.XGBClassifier(), 
                #"CatBoost": cat.CatBoostClassifier(),
                #"AdaBoost": AdaBoostClassifier(),
                #"LightGBM": lgb.LGBMClassifier(force_row_wise=True),
                }
            
            if all_parameters:
            # Finding the best parameters.
                params = { 
                    "RandomForest": {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                        },
                    "XGBoost": {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'n_estimators': [100, 200, 300]
                        },
                    "CatBoost": {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'depth': [3, 5, 7],
                        'iterations': [100, 200, 300]
                        },
                    "LightGBM" : {
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'num_leaves': [50, 100, 200]
                        }
                    }
            else:
            # I have the best params for XGBoost and RandomForest. Let's keep it simple
                params = {
                    "XGBoost": {
                        "objective": "multiclass",          # Objective function for the model
                        "metric": "multi_logloss",          # Evaluation metric
                        "verbosity": 0,                    # Verbosity level (-1 for silent)
                        "boosting_type": "gbdt",            # Gradient boosting type
                        "random_state": 42,       # Random state for reproducibility
                        "num_class": 7,                     # Number of classes in the dataset
                        'learning_rate': 0.030962211546832760,  # Learning rate for gradient boosting
                        'n_estimators': 500,                # Number of boosting iterations
                        'lambda_l1': 0.009667446568254372,  # L1 regularization term
                        'lambda_l2': 0.04018641437301800,   # L2 regularization term
                        'max_depth': 10,                    # Maximum depth of the trees
                        'colsample_bytree': 0.40977129346872643,  # Fraction of features to consider for each tree
                        'subsample': 0.9535797422450176,    # Fraction of samples to consider for each boosting iteration
                        'min_child_samples': 26             # Minimum number of data needed in a leaf
                    }

                }

            
            model_report:dict = evaluate_model(
                X_train = X_train, 
                y_train = y_train, 
                X_test = X_test,
                y_test = y_test, 
                models=models,
                params=params)
            
            logging.info("Model Evalaution is Completed.")
            logging.info(f"model metrics : {model_report}")

            # code differs for cross validation
            model = models['XGBoost']

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = model
            )

            return model_report  
        
        except Exception as e:
            raise CustomException(e, sys)

