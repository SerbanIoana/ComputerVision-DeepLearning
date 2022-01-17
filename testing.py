from pickle import load

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

from data_loading import load_data_set
from training import max_length_description, best_model_path

import numpy as np


def word_for_id(prediction, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            return word
    return None


def generate_description(model, tokenizer, photo, max_len):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_len):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_len)
        # predict next word
        prediction = model.predict([photo, sequence], verbose=0)
        # convert probability to integer==id of highest_probability
        prediction = np.argmax(prediction)
        # map integer to word
        word = word_for_id(prediction, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def evaluate_model(model, descriptions, photos, tokenizer, max_len):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        prediction = generate_description(model, tokenizer, photos[key], max_len)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(prediction.split())
    # calculate BLEU score
    print('BLEU-1: ', corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: ', corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: ', corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: ', corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == "__main__":
    # load test data
    test_descriptions, test_features = load_data_set('test', 'dataset\Flickr8k_text\Flickr_8k.testImages.txt')
    print(f'Descriptions: test={len(test_descriptions)}')
    print(f'Photos: test={len(test_features)}')

    tokenizer = load(open('tokenizer.pkl', 'rb'))

    model_filename = best_model_path
    model = load_model(model_filename)
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length_description)
