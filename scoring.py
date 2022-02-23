from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 

base_path = os.getcwd()
final_model_folder_path = os.path.join(base_path, model_path)
final_model_path = os.path.join(final_model_folder_path, 'trainedmodel.pkl')
test_data_folder_path = os.path.join(base_path, test_data_path)
test_data_complete_path = os.path.join(test_data_folder_path, 'testdata.csv')
scores_path = os.path.join(final_model_folder_path, 'latestscore.txt')

# Load model
with open(final_model_path, 'rb') as model_input:
        model = pickle.load(model_input)

#################Function for model scoring
def score_model(model):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Prep test data
    test_data = pd.read_csv(test_data_complete_path)
    columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
    final_test_data = test_data[columns_to_use]
    X = final_test_data.drop('exited', axis=1)
    y = final_test_data['exited']

    # Test model
    predictions = model.predict(X)
    f1_value = metrics.f1_score(y, predictions)

    # Save score 
    with open(scores_path, 'w') as output:
        output.write(str(f1_value))
    
    return f1_value

if __name__ == '__main__':
    score_model(model)