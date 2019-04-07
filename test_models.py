import json
from time import time
import logging
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 0 en
# 1 es
# 2 ar
def test_all_combinations(list_of_dataframes, list_of_labels, list_of_ids, models_dir='./models/'):
	models_files_paths = [join(models_dir, f) for f in listdir(models_dir) if isfile(join(models_dir, f))]
	loaded_models = [joblib.load(model_path) for model_path in models_files_paths]
		
	for i in range(1, 2 ** len(list_of_dataframes)):
		subset_of_dataframes = []
		subset_of_labels = []
		subset_of_ids = []
		for j in range(len(list_of_dataframes)):
			if (i & 1 << j) != 0:
				subset_of_ids.append(list_of_ids[j])
				subset_of_dataframes.append(list_of_dataframes[j])
				subset_of_labels.append(list_of_labels[j])
		if len(subset_of_dataframes) > 1:
			X = pd.concat(subset_of_dataframes, ignore_index=True)
			y = []
			for lab in subset_of_labels:
				y += lab
		else:
			X = subset_of_dataframes[0]
			y = subset_of_labels[0]
		t_start = time()
		for model_idx in range(len(loaded_models)):
			models_split = models_files_paths[model_idx].replace(models_dir, "") 
			models_split = models_files_paths[model_idx].strip().split('_')
			ids_models = []
			for s in models_split:
				if s == "svm":
					break
				ids_models.append(s)
			print("Using trained model {}".format(ids_models))
			print("Testing using {} languages".format(subset_of_ids))
			print("Shape of X dataframe: {}".format(X.shape))
			print("Shape of y dataframe: {}".format(len(y)))
			y_pred = loaded_models[model_idx].predict(X)
			print("Matrix of confusion")
			print(confusion_matrix(y, y_pred))
			print(classification_report(y, y_pred))
			print("Accuracy {}".format(accuracy_score(y, y_pred)))
			print("{} seconds to test the SVM classifier".format(time() - t_start))

def create_avg_dataset(dataframe, json_labels_file_path):
	X = dataframe.groupby('author_id').mean()
	with open(json_labels_file_path) as json_file:
		dict_of_labels = json.loads(json_file.read())
	y = [dict_of_labels[author_id] for author_id, row in X.iterrows()]
	return X, y

def main():
	test_data_csv_paths = ['./resnet_datasets/en_test.csv', './resnet_datasets/es_test.csv', './resnet_datasets/ar_test.csv']
	test_labels_json_paths = ['./authors_labels/en_test_labels.json', './authors_labels/es_test_labels.json',\
								'./authors_labels/ar_test_labels.json']
	X_dataframes = [None for i in range(len(test_data_csv_paths))]
	y_dataframes = [None for i in range(len(test_labels_json_paths))]
	list_of_ids = [None for i in range(len(X_dataframes))]
	i = 0
	t_start = time()
	for path_x, path_y in zip(test_data_csv_paths, test_labels_json_paths):
		list_of_ids[i] = path_x.strip().split('/')[2][:2]
		X = pd.read_csv(path_x, sep='\s*,\s*', header=0, encoding='ascii', engine='python')
		X_dataframes[i], y_dataframes[i] =  create_avg_dataset(X, path_y)
		i += 1
	print("{} seconds to read csv's and create all dataframes".format(time() - t_start))
	test_all_combinations(X_dataframes, y_dataframes, list_of_ids, './models/resnet/')

if __name__ == '__main__':
	main()