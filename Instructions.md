## End to End Machine Learning Project for Obesity Risk Prediction

Step 1: GitHub and Code Set Up

1. Setup the GitHub Repository
    a. new Environment
    b. setup.py
    c. requirements.txt

Setting Up Environemnt
1. Creating Environment 
    conda create -p venv python==3.8
2. Activate Environment - Activate from base
    conda activate venv/

Steps to Initialize Git Repo:
    ```
    git init
    git add README.md
    git commit -m "first commit"
    git branch -M main
    git remote add origin https://github.com/arunnandam/Obesity-Risk-Prediction.git
    git push -u origin main
    ```

Create setup.py file:
- With the help of setup.py, we can build our entire Application as a package and deploy it in PyPi.
- Create __init__.py in src folder and entire application is built in that folder. find_packages() will look for all the packages required for the project in __init__.py. Entire src is considered as package and used in all files.

Create requirements.txt file:
- requirements.txt has all the packages required to run the application.
- While running requirements.txt, we want to run the setup.py also, for this we need to add <strong>e .</strong> at the end and handle it in setup.py to avoid error in install_requires.
```
pip install -r requirements.txt
```

- At the end [text](Obesity_Risk_Prediction.egg-info) will be created.

Step 2: Creating Project Structure, Logging and Exceptions

1. Create <i>components</i> folder in src. components are like all the modules used in the project such as Data Ingestion, Data Transformation, Model Training, 
2. Create <i>pipeline</i> folder in src. pipelines includes the training and prediction pipeline.
3. Create logger.py to record all the logs
4. Create exception.py to handle all exceptions.
5. Create utils.py to save the functionalities written in a common way. Example, to save the model in the cloud.

Step 3: Project Problem Statement, EDA and Model Building

1. Doing Exploratory Data Analysis of all Columns and Finding the hidden insights.
2. Done Feature Engineering 
3. Created Transformers to encode and scale the variables
4. Tested across different clsssification models and tested the metrics.
5. Cross Validation and HyperParameter Tuning.
6. Plotting Predictions Distribution.

Step 4: Creating Data Ingestion 

- The purpose of data ingestion is to create ingestion class that reads our data from given source and return the paths. We can also create artifacts folder, so that they will be stored.

1. Create DataIngestionConfig and DataIngestion classes 
2. DataIngestionConfig - a dataclass that defines the paths of the train, test, raw_data
3. DataIngestion -  To initiate the data ingestion process
4. Logging and exceptions are loaded to log the info and track the custom exceptions.

Step 5: Creating Data Transformation.

- The purpose of Data Transfromation is to transform all the numerical and categorical columns and save the pickle file.

1. Create DataTransformationConfig and DataTransformation classes.
2. Create DataTransformer object by specifying the numerical and categorical pipelines. Finally create ColumnTransfromer and return it.
3. Initiate Data Transformation and fit the pipelines for both train and test data.
4. return the preprocessed arrays and save the preprocessor object.
5. create utils.py to write all the usable code in the project

Step 6: Creating Model Trainer and Hyper parameter Tuning

- The purpose of Model Trainer is to train the data across models and return the model evaluation report

1. Create ModelTrainer and ModelTrainerConfig classes.
2. split the train and test data. Create a function in utils that will trains the model and return the dict of kpis
3. return the model_report and export it.
4. Do the Hyper parameter tuning for the model.

Step 7: Creating Prediction pipeline
- In this step, I will create the prediction pipeline, the flask app and home.html

1. The flask app accepts the input from the user and all the values are sent to predicton pipeline
2. The web page <italic>home.html</italic> takes values and extracted in <italic>app.py</italic> using request.get.values()
3. The data is then scaled and we will get the prediction from the stored pickle files.
4. The prediction is sent back to the form after mapping.
