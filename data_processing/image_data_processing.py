import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from pickle import dump

from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import Model

from keras_preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input


def square_image(img):
    width_pad = 0
    height_pad = 0
    # if img is taller
    if img.shape[0] > img.shape[1]:
        width_pad = (img.shape[0] - img.shape[1]) // 2
    else:  # if img is wider
        height_pad = (img.shape[1] - img.shape[0]) // 2
    padded_image = np.pad(img, ((height_pad, height_pad), (width_pad, width_pad), (0, 0)), mode="edge")
    return padded_image


def process_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = square_image(img)
    img = cv2.resize(img, (224, 224))
    # reshape data for the model, expand the dimensions to be (1, 3, 224, 224)
    img = np.expand_dims(img, axis=0)
    # prepare the img for the VGG model
    img = preprocess_input(img)
    return img


# extract features from each photo in the dir
def extract_features(dir):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract img_features from each photo
    img_features = dict()
    for name in listdir(dir):
        # load an img from file
        filename = dir + '/' + name
        img = process_image(filename)

        # get features
        feature = model.predict(img, verbose=0)
        # get img id
        image_id = name.split('.')[0]
        # store feature
        img_features[image_id] = feature
        print(f'>>>{name}\n')
    return img_features


if __name__ == "__main__":
    # extract features from all images
    directory = os.path.dirname(__file__) + '\..\dataset\Flickr8k_Dataset'
    print(directory)

    print(f'Begin extracting features from: {directory}')
    features = extract_features(directory)
    print(f'Extracted features: {len(features)}')
    # save to file
    dump(features, open('../resources/img_features.pkl', 'wb'))
