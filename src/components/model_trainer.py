# All the model training code

# Importing the packages

import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from dataclasses import dataclass

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
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    '''
    Defining the model pickle path
    '''
    trained_model_file_path : str=os.path.join('artifacts','model.pkl')

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
                "DecisonTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "XGBoost": xgb.XGBClassifier(), 
                "CatBoost": cat.CatBoostClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "LightGBM": lgb.LGBMClassifier(force_row_wise=True),
                }

            model_report:dict = evaluate_model(
                X_train = X_train, 
                y_train = y_train, 
                X_test = X_test,
                y_test = y_test, 
                models=models)
            
            logging.info("Model Evalaution is Completed.")
            logging.info(f"model metrics : {model_report}")

            return model_report  
        
        except Exception as e:
            raise CustomException(e, sys)

