import json
import pickle
import os
import ast
import ingestion
import training
#import scoring
import deployment
import diagnostics
import reporting
import apicalls
import pandas as pd
from sklearn import metrics

##################Check and read new data
with open('config.json','r') as f:
    config = json.load(f) 
base_path = os.getcwd()
ingested_path = os.path.join(base_path, config['output_folder_path'], 'ingestedfiles.txt')
data_path = os.path.join(base_path, config['output_folder_path'], 'finaldata.csv')
source_data_path = os.path.join(base_path, config['input_folder_path'])
scores_path = os.path.join(base_path, config['prod_deployment_path'], 'latestscores.txt')
model_path = os.path.join(base_path, config['prod_deployment_path'], 'trainedmodel.pkl')
#first, read ingestedfiles.txt
with open(ingested_path, 'r') as f:
    ingested_files = ast.literal_eval(f.read())


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = os.listdir(source_data_path)

unloaded_data = [filename for filename in source_files if filename not in ingested_files]


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if unloaded_data != None:
    ingestion

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(scores_path, 'r') as scores_output:
    scores = scores_output.read()
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
# Test model on data
test_data = pd.read_csv(data_path)
columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
final_test_data = test_data[columns_to_use]
X = final_test_data.drop('exited', axis=1)
y = final_test_data['exited']
predictions = model.predict(X)
f1_value = metrics.f1_score(y, predictions)

if float(f1_value) < float(scores):
    print('Retraining needed')
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
training
#scoring
#################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
diagnostics
reporting
apicalls








