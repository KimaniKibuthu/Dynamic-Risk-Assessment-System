import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

base_path = os.getcwd()
full_input_folder_path = os.path.join(base_path, input_folder_path)
full_output_folder_path = os.path.join(base_path, output_folder_path)
final_data_path = os.path.join(full_output_folder_path, 'finaldata.csv')
final_files_path = os.path.join(full_output_folder_path, 'ingestedfiles.txt')

if not os.path.exists(full_output_folder_path):
    os.makedirs(full_output_folder_path)


def text_saver(filepath, values):
    with open(filepath, 'w') as output:
        output.write(str(values))


# Function for data ingestion
def merge_multiple_dataframe():
    # Check for datasets
    # Specify directories and create output directory
    files = os.listdir(full_input_folder_path)
    if files is None:
        print('No files found')
    else:
        final_data = pd.DataFrame()
        files_read = []
        for file in files:
            file_path = os.path.join(full_input_folder_path, file)
            file_extension = file.split('.')[-1]
            if file_extension == 'csv':
                files_read.append(file)
                data = pd.read_csv(file_path)
                final_data = final_data.append(data)
        # Save merged to one csv file & files to txt
        final_data = final_data.drop_duplicates()
        final_data.to_csv(final_data_path, index=False)
        text_saver(final_files_path, files)


if __name__ == '__main__':
    merge_multiple_dataframe()
