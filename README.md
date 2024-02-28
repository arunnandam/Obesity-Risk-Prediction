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