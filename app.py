from flask import Flask, request, render_template, url_for, app
from flask.helpers import make_response
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import cols, num_cols
from src.logger import logging

#entry point 
application=Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predicting data
@app.route('/predict_data', methods=['GET','POST'])
def predict_data():
    k = list(request.form.values())
    k1 = request.form.get('Gender')
    logging.info(k1)
    if request.method=='GET':
        return render_template('home.html', result = "")
    else:
        
        args = []
        for i in range(len(cols)):
            if cols[i] in num_cols:
                args.append(float(request.form.get(cols[i])))
            else:
                args.append(request.form.get(cols[i]))

        data = CustomData(*args)

        prediction_df = data.get_data_as_dataframe()
        logging.info(prediction_df)

        predict_pipeline = PredictPipeline()
        result_df, mapping = predict_pipeline.predict(prediction_df)
        
        return render_template('home.html', result = mapping[result_df[0]])
    
if __name__=="__main__":
    app.run(debug=True)


     





