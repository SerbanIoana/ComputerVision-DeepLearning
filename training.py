from pickle import dump

from domain.data_generator import data_generator
from data_loading import *
from domain.model import define_model

max_length_description = 0
best_loss = 0
best_model_path = ''

if __name__ == "__main__":
    # load training dataset (6K)
    train_descriptions, train_features = load_data_set('train', '\dataset\Flickr8k_text\Flickr_8k.trainImages.txt')
    print(f'Descriptions: train={len(train_descriptions)}')
    print(f'Photos: train={len(train_features)}')

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    dump(tokenizer, open('resources\\tokenizer.pkl', 'wb'))
    print(f'Vocabulary Size: {vocab_size}')

    # determine the maximum sequence length
    max_length_description = max_desc_length(train_descriptions)
    print(f'Description max length: {max_length_description}')

    # define the model
    model = define_model(vocab_size, max_length_description)

    # train the model, run iterations manually and save after each epoch
    iterations = 10
    batch_size = 300
    steps = len(train_descriptions) // batch_size
    for i in range(iterations):
        # create the data generator
        generator = data_generator(train_descriptions, train_features, tokenizer, max_length_description, vocab_size, batch_size)
        # compile with loss metrics
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
        # fit for one epoch
        history = model.fit(generator, epochs=5, steps_per_epoch=steps, verbose=1)
        # save model
        loss = history.history['loss']
        model_path = 'train_models\model_ep' + str(i) + '_loss' + ''.join(str(loss).split('.')) + '.h5'

        if i == 0:
            best_loss = loss
            best_model_path = model_path
        else:
            if loss < best_loss:
                best_loss = loss
                best_model_path = model_path

        model.save(model_path)