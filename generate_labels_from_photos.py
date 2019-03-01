import os
import json
import sys

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


model = VGG16()
BASE_DIR_PHOTOS = sys.argv[1]
image_labels_path = sys.argv[2]

def get_list_of_labels(image_path):
	image = load_img(image_path, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	yhat = model.predict(image)
	labels = decode_predictions(yhat)
	return [(label[1], str(label[2])) for label in labels[0]]

def get_authors_dirs(path):
	dirs = os.listdir(path)
	return [dir_name for dir_name in dirs if os.path.isdir(path + '/' + dir_name)]


def label_author_photos(author_dir_path):
	author_images_json = {}
	list_of_files = os.listdir(BASE_DIR_PHOTOS + '/' + author_dir_path)
	for image_file in list_of_files:
		labels = get_list_of_labels(BASE_DIR_PHOTOS + '/' + author_dir_path + '/' + image_file)
		author_images_json[image_file] = labels
	with open(image_labels_path + '/' + author_dir_path + ".json", 'w') as fp:
	    json.dump(author_images_json, fp)

def main():
	authors = get_authors_dirs(BASE_DIR_PHOTOS)
	for author in authors:
		label_author_photos(author)

if __name__ == '__main__':
	main()