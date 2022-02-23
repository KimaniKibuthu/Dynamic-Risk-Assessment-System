from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import shutil
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_data_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

base_path = os.getcwd()
ingested_files_path = os.path.join(base_path, output_data_path, 'ingestedfiles.txt')
trained_model_path = os.path.join(base_path, model_path, 'trainedmodel.pkl')
scores_path = os.path.join(base_path, model_path, 'latestscore.txt')
deployment_folder_path = os.path.join(base_path, prod_deployment_path)

if not os.path.exists(deployment_folder_path):
    os.makedirs(deployment_folder_path)
# Create files in deployment folder

####################function for deployment
def store_model_into_pickle():
  
    shutil.copy(trained_model_path, os.path.join(deployment_folder_path, 'trainedmodel.pkl'))
    shutil.copy(scores_path, os.path.join(deployment_folder_path, 'latestscores.txt'))
    shutil.copy(ingested_files_path, os.path.join(deployment_folder_path, 'ingestedfiles.txt'))

if __name__ == '__main__':
    store_model_into_pickle()  
        

