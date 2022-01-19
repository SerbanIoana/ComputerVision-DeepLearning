import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add

# define the captioning model
from keras.utils.vis_utils import plot_model


def define_model(vocab_size, max_length):
    # feature extractor model
    f1 = Input(shape=(2048,))
    f = Dropout(0.5)(f1)
    f = Dense(256, activation='relu')(f)
    # sequence model
    s1 = Input(shape=(max_length,))
    s = Embedding(vocab_size, 256, mask_zero=True)(s1)
    s = Dropout(0.5)(s)
    s = LSTM(256)(s)
    # decoder model
    d = add([f, s])
    d = Dense(256, activation='relu')(d)
    d = Dense(vocab_size, activation='softmax')(d)
    # tie it together [image, seq] [word]
    model = Model(inputs=[f1, s1], outputs=d)
    # compile model
    return model


if __name__ == "__main__":
    model = define_model(7579, 34)
    # summarize model
    model.summary()
    tf.keras.utils.plot_model(model, to_file='../resources/model.png', show_shapes=True)
