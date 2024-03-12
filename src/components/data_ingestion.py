import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# Paths for all data class[ raw data, transformed data ]
@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts','train.csv') 
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','raw_data.csv')
    final_test_path: str=os.path.join('artifacts','final_test.csv')

# If you want to define some variables, go with dataclass. Otherwise write __init__
# dataclass adds decorators automatically and eliminates writing __init__, __hash__ etc., functions
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Write code to read the datasets
        '''

        logging.info("Entered the data ingestion method")
        try:
            traindf = pd.read_csv('notebook/data/train.csv')
            testdf = pd.read_csv('notebook/data/test.csv')
            logging.info('Read both test and train data.')

            # In this dataset, test data has no target variable, so we train and test our entire model on traindf
            # let's store testdf as final_test and import at last to make predictions.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            traindf.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data saved.")

            logging.info("Splitting the data into train and test")
            train_df, test_df = train_test_split(traindf, test_size=0.2, random_state=42)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Storing final_test data
            testdf.to_csv(self.ingestion_config.final_test_path, index=False, header=True)


            logging.info("Ingestion of the data is completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_df, test_df = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, mapping, _ = data_transformation.initiate_data_transformation(train_df, test_df)

    model_trainer = ModelTrainer()
    model_report = model_trainer.initiate_model_trainer(train_arr, test_arr, mapping, _)

    model_report_df = pd.DataFrame(model_report).T
    model_report_df.columns = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1']
    model_report_df.to_csv('model_evaluation_report.csv', header=True, index=False)



