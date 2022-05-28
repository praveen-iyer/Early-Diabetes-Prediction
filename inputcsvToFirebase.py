import pandas as pd
import json
import requests

path='diabetes_data_upload.csv'
baseURL='https://dsci551-c8eb7-default-rtdb.firebaseio.com/.json'

def read_csv(path):
	print('Reading csv data and converting into a dataframe')
	return pd.read_csv(path)

def df_to_json(dataframe):
	print('Converting dataframe into json format')
	output_json=dataframe.to_json(orient='index')
	return json.loads(output_json)
		
def clear_database(baseURL):
	print('Deleting pre-existing records from the database')
	response=requests.delete(url=baseURL)
		
def append_data(baseURL,json_data):
	print('Uploading the json format into Firebase')
	response=requests.put(url=baseURL,json=json_data)
	print('Uploaded')
	
if __name__ == "__main__":
	diabetes_data=read_csv(path)
	diabetes_data_json=df_to_json(diabetes_data)
	clear_database(baseURL)
	append_data(baseURL,diabetes_data_json)