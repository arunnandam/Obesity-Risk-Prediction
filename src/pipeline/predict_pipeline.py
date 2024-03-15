# Importing the required packages
import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths of pickle files
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            mapping_path = 'artifacts/mapping.pkl'

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            mapping = load_object(file_path=mapping_path)

            logging.info(f"preprocessor:{preprocessor} model:{model}")

            # Transforming the features
            logging.info(features)
            scaled_data = preprocessor.transform(features)

            # Pfredicting for given data
            prediction = model.predict(scaled_data)

            return prediction, mapping
        
        except Exception as e:
            raise CustomException(e, sys)




class CustomData:
    def __init__(self, *args):
        # assign to inputs
        self.cols = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
        for i in range(len(args)):
            logging.info(f"column:{self.cols[i]} value:{args[i]}")
            setattr(self, self.cols[i], args[i])

        
    def get_data_as_dataframe(self):
        try:
            #cols = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
            custom_data_input_dict = {}
            
            for i in range(len(self.cols)):
                custom_data_input_dict[self.cols[i]] = [getattr(self,self.cols[i])]

            logging.info(custom_data_input_dict)

            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e, sys)






