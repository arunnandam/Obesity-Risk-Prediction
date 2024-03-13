import logging
import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
CV = False # Set it to true in case of HP and CV




def save_object(file_path, obj):
    '''
    saving the pickle files
    '''

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def kpis(y_true, y_pred):
    '''
    Finding Metrics of the data and sending an dataframe with accuracy, precision, recall, f1 score.
    '''
    # calculate accuracy for test set
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate weighted average precision and recall
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Calculate macro-averaged precision and recall
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return [accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1] 

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    model_names =list(models.keys())
    model_func = list(models.values())
    final_metrics = {}
    
    for i in range(len(model_names)):
        model = model_func[i]
        param = params[model_names[i]]

        logging.info(f"model:{model} parameters: { param }")

        if CV:
            gridSearch = GridSearchCV(model, param, cv=3)
            gridSearch.fit(X_train, y_train)

            #Selecting the best parameters
            best_params = gridSearch.best_params_
            logging.info(f"model:{model_names[i]} params:{best_params}")
            model.set_params(**best_params)
        else:
            model.set_params(**param)
    
        #Training the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate Metrics
        metrics = kpis(y_test, y_pred)

        final_metrics[model_names[i]] = metrics
        logging.info(metrics)
        
    return final_metrics