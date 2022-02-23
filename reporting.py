import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

base_path = os.getcwd()
test_data_path = os.path.join(base_path, config['test_data_path'], 'testdata.csv')
model_path =  os.path.join(base_path, config['prod_deployment_path'], 'trainedmodel.pkl')




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    with open(model_path, 'rb') as model_input:
        model = pickle.load(model_input)
    test_data = pd.read_csv(test_data_path)
    columns_to_use = [column for column in test_data.columns if test_data[column].dtype != 'O']
    final_test_data = test_data[columns_to_use]
    X = final_test_data.drop('exited', axis=1)
    y = final_test_data['exited']

    # Test model
    predictions = model.predict(X)
    cm = confusion_matrix(y, predictions)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    cm_display
    plt.savefig(os.path.join(base_path, config['output_model_path'], 'confusion_matrix.png'))


if __name__ == '__main__':
    score_model()
