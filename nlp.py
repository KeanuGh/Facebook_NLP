import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    vocab = {start_token, end_token}  # add stop and start tokens
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


def gen_input_and_target(text, column_name, seq_length, char_to_idx, batch_size=64, buffer_size=10000, pickle_filename=None):
    """
    TODO
    :param text:
    :param column_name:
    :param seq_length:
    :param char_to_idx:
    :param batch_size:
    :param buffer_size:
    :param pickle_filename:
    :return:
    """
    # get max length
    max_len = text[column_name].map(len).max()
    print(text[column_name])
    # doing the input and targets myself because I am confused
    # create column in dataframe containing vector rep. of characters
    texts_as_int = [[char_to_idx[c] for c in seq] for seq in text[column_name]]
    # pad sequences to max length (after text)
    texts_as_int = pad_sequences(texts_as_int, max_len, padding='post')

    # creates tensorflow dataset object from text as integers
    char_dataset = tf.data.Dataset.from_tensor_slices(texts_as_int)
    # creates sequences of size seq_length + 1
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    # inputs are chunks of length seq_length, targets are the next character
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[-1]
        return input_text, target_text
    dataset = sequences.map(split_input_target)
    print(f'dataset pre-batched: {dataset}')
    # shuffle dataset to buffer size
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print(f'dataset batched: {dataset}')

    # print file to pickle
    if pickle_filename:
        with open(pickle_filename, 'wb') as file:
            pickle.dump(pickle_filename, file)
        print(f"printed dataset to file {pickle_filename}")

    return dataset


def generate_model(vocab, rnn_units=50, embedding_dim=256, batch_size=64):
    """
    TODO
    :param vocab:
    :param rnn_units:
    :param embedding_dim:
    :param batch_size:
    :return:
    """
    model = Sequential()
    model.add(Embedding(len(vocab), embedding_dim, batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_units, return_sequences=True,
                  dropout=0.5, recurrent_dropout=0.15,
                  stateful=True,
                  recurrent_initializer='glorot_uniform'))
    # model.add(GRU(rnn_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.15,
    # recurrent_initializer='glorot_uniform'))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model
