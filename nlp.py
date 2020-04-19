import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# disables AVX/FMA warning as I'm using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def start_and_end_tokens(texts, start_token='\t', end_token='\n'):
    """
    Adds token (default '\t') to the start of text
    adds token (default '\n') to the end of text
    :param texts: pandas Series containing texts
    :param start_token: start token
    :param end_token: end token
    :return: pandas series containing texts with start and end tokens
    """
    texts_with_bookends = texts.apply(lambda x: start_token + x + end_token)
    return texts_with_bookends


def build_vocabulary(texts, start_token='\t', end_token='\n'):
    """
    builds a set containing every unique character in the texts
    :param end_token: end token
    :param start_token: start token
    :param texts: pandas Series containing all texts
    :return: vocabualary set
    """
    vocab: set = {start_token, end_token}  # add stop and start tokens
    for text in texts:
        for c in text:
            if c not in vocab:
                vocab.add(c)
    return vocab


def char_to_int_maps(vocab):
    """
    builds two dictionaries to map from numerical to characters and back
    :param vocab: set of vocabulary
    :return: tuple (character-to-index dictionary, index-to-character dictionary)
    """
    char_to_idx = {char: idx for idx, char in enumerate(sorted(vocab))}
    idx_to_char = {idx: char for idx, char in enumerate(sorted(vocab))}
    return char_to_idx, idx_to_char


def gen_input_and_target(text, char_to_idx, vocab: set, seq_length=20, step=1, pickle_filename=None):
    """
    TODO
    :param vocab:
    :param step:
    :param text:
    :param seq_length:
    :param char_to_idx:
    :param pickle_filename:
    :return: dataframe w/ columns 'inputs', 'targets' containing text sequences and their next
    """
    # get max length
    max_len = text.map(len).max()
    print(text)
    # doing the input and targets myself because I am confused
    # create column in dataframe containing vector rep. of characters
    texts_as_int = [[char_to_idx[c] for c in seq] for seq in text]
    # pad sequences to max length (after text)
    texts_as_int = pad_sequences(texts_as_int, max_len, padding='pre')

    # instantiate lists to contain sequences and next characters
    texts = []
    next_chars = []
    # loop over each text in texts to get subset of length seq_length and the next character
    for text in texts_as_int:
        for i in range(0, len(text) - seq_length, step):
            texts.append(text[i:i + seq_length])
            next_chars.append(text[i + seq_length])
    # create dataframe with these lists ad columns
    ml_data = pd.DataFrame({'inputs': texts, 'targets': next_chars})
    # instanciate empty vectors to contain input and target data
    numerical_texts = np.zeros((len(ml_data['inputs']), seq_length + 1, len(vocab)), dtype=bool)
    numerical_next_chars = np.zeros((len(ml_data['targets']), len(vocab)), dtype=bool)
    # vectorise imput and targets
    for i, text in enumerate(ml_data['inputs']):
        for t, idx in enumerate(text):
            numerical_texts[i, t, idx] = 1
            numerical_next_chars[i, idx] = 1

    # print file to pickle
    if pickle_filename:
        with open(pickle_filename, 'wb') as file:
            pickle.dump(ml_data, file)
        print(f"printed dataset to file {pickle_filename}")

    return numerical_texts, numerical_next_chars


def generate_model(vocab, seq_len, rnn_units=50):
    """
    TODO
    :param seq_len:
    :param vocab:
    :param rnn_units:
    :param embedding_dim:
    :return:
    """
    model = Sequential()
    # model.add(Embedding(len(vocab), embedding_dim,
    #                    mask_zero=True))
    model.add(GRU(rnn_units, input_shape=(seq_len + 1, len(vocab)),
                  return_sequences=True,
                  dropout=0.15, recurrent_dropout=0.15,
                  recurrent_initializer='glorot_uniform'))
    model.add(GRU(rnn_units, return_sequences=False,
                  recurrent_initializer='glorot_uniform'))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model


def fit_model(model, inputs, targets):
    # create checkpoints for callbacks
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # fit model
    history = model.fit(inputs, targets,
                        epochs=10,
                        callbacks=[checkpoint_callback],
                        batch_size=64,
                        verbose=2)
    model.reset_metrics()
    model.save('model.h5')
    return model


def generate_text(model, n: int, max_len: int, vocab: set, char_to_idx: dict, idx_to_char: dict,
                  start_token: str = '\t', end_token: str = '\n'):

    # generate n texts
    for i in range(0, n):
        stop = False
        counter = 1
        text = ''

        # to contain output text, initialise by filling with start character
        output_seq = np.zeros([1, max_len + 1, len(vocab)])
        output_seq[0, 0, char_to_idx[start_token]] = 1.

        # generate new characters until you reach end token or text reaches maximum length
        while not stop and counter < max_len + 1:
            probs = model.predict_proba(output_seq, verbose=0)[0]
            c = np.random.choice(sorted(list(vocab)), replace=True, p=probs.reshape(len(vocab)))
            if c == end_token:
                stop = True
            else:
                text += c
                output_seq[0, counter, char_to_idx[c]] = 1.
                counter += 1
        print(text)


if __name__ == '__main___':
    exit(0)
