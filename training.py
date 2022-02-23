from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

# Create models folder
base_path = os.getcwd()
final_model_folder_path = os.path.join(base_path, model_path)
final_model_path = os.path.join(final_model_folder_path, 'trainedmodel.pkl')
data_folder_path = os.path.join(base_path, dataset_csv_path)
data_path = os.path.join(data_folder_path, 'finaldata.csv')

if not os.path.exists(final_model_folder_path):
    os.makedirs(final_model_folder_path)

#################Function for training the model
def train_model():
    # Read data
    data = pd.read_csv(data_path)
    # Split the data
    columns_to_use = [column for column in data.columns if data[column].dtype != 'O']
    train_data = data[columns_to_use]
    X = train_data.drop('exited', axis=1)
    y = train_data['exited']

    #use this logistic regression for training
    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    lr_model.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(final_model_path, 'wb') as model_output:
        pickle.dump(lr_model, model_output)

if __name__ == '__main__':
    train_model()