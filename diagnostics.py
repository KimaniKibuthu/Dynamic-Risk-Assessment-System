
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import training
import ingestion
import subprocess
from itertools import chain

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f) 

production_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path'])
ingested_data_path = os.path.join(config['output_folder_path'])

base_path = os.getcwd()
model_path = os.path.join(base_path, production_path, 'trainedmodel.pkl')
test_data_complete_path = os.path.join(base_path, test_data_path, 'testdata.csv')
final_data_path = os.path.join(base_path, ingested_data_path, 'finaldata.csv')


# Function to get model predictions
def model_predictions():
    # read the deployed model and a test dataset, calculate predictions
    with open(model_path, 'rb') as model_input:
        model = pickle.load(model_input)
    test_data = pd.read_csv(test_data_complete_path)
    columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
    final_test_data = test_data[columns_to_use]
    X = final_test_data.drop('exited', axis=1)
    y = final_test_data['exited']

    # Test model
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    return predictions


def missing_data_checker():
    data = pd.read_csv(final_data_path)
    percentage_of_missing = [data[column].isnull().sum()/len(data) for column in data]
    return percentage_of_missing


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    data = pd.read_csv(final_data_path)
    average = list(data.mean())
    medians = list(data.median())
    stds = list(data.std())
    summary = list(chain(average, medians, stds))

    return summary


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    ingestion_start = timeit.default_timer()
    ingestion
    ingestion_end = timeit.default_timer()
    ingestion_timing = ingestion_end - ingestion_start

    training_start = timeit.default_timer()
    training
    training_end = timeit.default_timer()
    training_timing = training_end - training_start
    return [ingestion_timing, training_timing]


# Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    outdated = subprocess.check_output(['pip', 'list', '--outdated'])
    with open('outdated.txt', 'wb') as f:
        f.write(outdated)


if __name__ == '__main__':
    model_predictions()
    missing_data_checker()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
