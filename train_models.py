import json
import logging
from time import time
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# 0 en
# 1 es
# 2 ar

def train_all_combinations(list_of_dataframes, list_of_labels, list_of_ids, models_dir='./models/'):
	for i in range(1, 2 ** len(list_of_dataframes)):
		subset_of_dataframes = []
		subset_of_labels = []
		subset_of_ids = []
		for j in range(len(list_of_dataframes)):
			if (i & 1 << j) != 0:
				subset_of_ids.append(list_of_ids[j])
				subset_of_dataframes.append(list_of_dataframes[j])
				subset_of_labels.append(list_of_labels[j])
		print(subset_of_ids)
		if len(subset_of_dataframes) > 1:
			X = pd.concat(subset_of_dataframes, ignore_index=True)
			y = []
			for i in subset_of_labels:
				y += i
		else:
			X = subset_of_dataframes[0]
			y = subset_of_labels[0]
		print("Shape of X dataframe: {}".format(X.shape))
		print("Shape of y dataframe: {}".format(len(y)))
		t_start = time()
		param = [{"C": [0.01, 0.1, 1, 10, 100, 1000]}]
		clf = GridSearchCV(LinearSVC(), param)
		clf.fit(X, y)
		print("Best parameter for classifier {}".format(clf.best_params_))
		print("{} seconds to train the SVM classifier".format(time() - t_start))
		joblib.dump(clf, '{}{}_svm_kernel_lin_avg_grid.sav'.format(models_dir, '_'.join(subset_of_ids)))

def create_avg_dataset(dataframe, json_labels_file_path):
	X = dataframe.groupby('author_id').mean()
	with open(json_labels_file_path) as json_file:
		dict_of_labels = json.loads(json_file.read())
	y = [dict_of_labels[author_id] for author_id, row in X.iterrows()]
	return X, y

def main():
	train_data_csv_paths = ['./resnet_datasets/en_train.csv', './resnet_datasets/es_train.csv', './resnet_datasets/ar_train.csv']
	train_labels_json_paths = ['./authors_labels/en_train_labels.json', './authors_labels/es_train_labels.json',\
								'./authors_labels/ar_train_labels.json']
	X_dataframes = [None for i in range(len(train_data_csv_paths))]
	y_dataframes = [None for i in range(len(train_labels_json_paths))]
	list_of_ids = [None for i in range(len(X_dataframes))]
	i = 0
	t_start = time()
	for path_x, path_y in zip(train_data_csv_paths, train_labels_json_paths):
		list_of_ids[i] = path_x.strip().split('/')[2][:2]
		X = pd.read_csv(path_x, sep='\s*,\s*', header=0, encoding='ascii', engine='python')
		X_dataframes[i], y_dataframes[i] =  create_avg_dataset(X, path_y)
		i += 1
	print("{} seconds to read csv's and create all dataframes".format(time() - t_start))
	train_all_combinations(X_dataframes, y_dataframes, list_of_ids, './models/resnet/')

if __name__ == '__main__':
	main()