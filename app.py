from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import f1_score
import diagnostics 
import json
import os
import scoring



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

base_path = os.getcwd()
dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path =  os.path.join(base_path, config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = os.path.join(base_path, config['test_data_path'], 'testdata.csv')

with open(model_path, 'rb') as model_input:
        prediction_model = pickle.load(model_input)



#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    test_data = pd.read_csv(test_data_path)
    columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
    final_test_data = test_data[columns_to_use]
    X = final_test_data.drop('exited', axis=1)
    y = final_test_data['exited']

    # Test model
    predictions = prediction_model.predict(X)
    return str(list(predictions))

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score_stats():        
    #check the score of the deployed model
    test_data = pd.read_csv(test_data_path)
    columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
    final_test_data = test_data[columns_to_use]
    X = final_test_data.drop('exited', axis=1)
    y = final_test_data['exited']

    # Test model
    predictions = prediction_model.predict(X)
    f1_value = f1_score(y, predictions)
    
    return str(f1_value)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stat_summary():        
    stat_summary = diagnostics.dataframe_summary()
    return str(stat_summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def other_stats():        
    #check timing and percent NA values
    timing = diagnostics.execution_time()
    missing = diagnostics.missing_data_checker()
    return str(timing), str(missing)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
