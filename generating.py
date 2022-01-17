from pickle import load

from keras.applications.vgg16 import VGG16
from keras.models import load_model, Model

from data_processing.image_data_processing import process_image
from testing import generate_description
from training import best_model_path, max_length_description


def extract_features_photo(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    img = process_image(filename)
    # get features
    features = model.predict(img, verbose=0)
    # return feature
    return features


if __name__ == "__main__":
    # load the tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # load the model
    model_filename = best_model_path
    model = load_model(model_filename)
    # load and prepare the photograph
    photo = extract_features_photo('example.jpg')
    # generate description
    description = generate_description(model, tokenizer, photo, max_length_description)
    print(description)