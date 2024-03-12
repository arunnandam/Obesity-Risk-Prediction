# All the data transformations

# Importing the packages
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, traindf, testdf):
        '''
        This function is responsible for data transformations for both numerical and categorical data
        '''
        try:
            logging.info("Data Transfromation initiated.")
            numerical_columns = traindf.select_dtypes(exclude="object").columns
            categorical_columns = traindf.select_dtypes(include="object").columns
            target_column = categorical_columns[-1]
            categorical_columns = categorical_columns.drop('NObeyesdad')

            # Numerical pipeline
            # Handle missing values and impute them.
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            # To handle the missing values and encoding
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical scaling and Categorical Columns encoding completed.")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            logging.info("Data Transformations completed.")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        To Initiate the data transformations
        '''

        try:
            traindf = pd.read_csv(train_path)
            testdf = pd.read_csv(test_path)

            logging.info("Reading data completed")

            logging.info("Obtaining preprocessor object.")

            preprocessor_obj = self.get_data_transformer_object(traindf, testdf)

            target_column = "NObeyesdad"

            input_features_traindf = traindf.drop(columns=[target_column], axis=1)
            target_feature_traindf = traindf[target_column]

            input_feature_testdf = testdf.drop(columns = [target_column], axis=1)
            target_feature_testdf = testdf[target_column]

            logging.info("Applying preprocessing on data")

            input_features_traindf_arr = preprocessor_obj.fit_transform(input_features_traindf)
            input_features_testdf_arr = preprocessor_obj.transform(input_feature_testdf)

            logging.info("Applying preprocessing on target variable")
            
            label_encoder = LabelEncoder()
            
            target_feature_traindf_arr = label_encoder.fit_transform(target_feature_traindf)
            target_feature_testdf_arr = label_encoder.fit_transform(target_feature_testdf)

            logging.info("Creating mapping after encoding the test data")
            # Creating mapping of target variable
            mapping_df = pd.DataFrame([target_feature_traindf,target_feature_traindf_arr]).T
            mapping_df = mapping_df.rename(columns={'Unnamed 0': 'NObeyesdad_encoded'})
            mapping_df = mapping_df.groupby(by = ['NObeyesdad','NObeyesdad_encoded']).size().reset_index()
            mapping = dict(zip(mapping_df['NObeyesdad_encoded'],mapping_df['NObeyesdad']))
            logging.info(f"mapping done : {mapping}")

            train_arr = np.c_[input_features_traindf_arr, target_feature_traindf_arr]
            test_arr = np.c_[input_features_testdf_arr, target_feature_testdf_arr]
            


            logging.info("Saving preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                mapping,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)


            

    