from os import getcwd
import requests
import subprocess
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f)
base_path = os.getcwd()
output_path = os.path.join(base_path, config['output_model_path'], 'apireturns.txt')
#Call each API endpoint and store the responses
# response1 = requests.get(f'{URL}/prediction').content
# response2 = requests.get(f'{URL}/scoring').content
# response3 = requests.get(f'{URL}/summarystats').content
# response4 = requests.get(f'{URL}/diagnostics').content

response1 = subprocess.run(['curl', f'{URL}/prediction'],capture_output=True).stdout
response2 = subprocess.run(['curl', f'{URL}/scoring'],capture_output=True).stdout
response3 = subprocess.run(['curl', f'{URL}/summarystats'],capture_output=True).stdout
response4 = subprocess.run(['curl', f'{URL}/diagnostics'],capture_output=True).stdout

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open(output_path, 'w') as output:
    output.write(str(responses))




