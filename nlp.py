import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


def start_and_end_tokens(texts: np.array, end_token: str = '\n') -> tuple:
    """
    adds token (default '\n') to the end of text
    """
    texts_stopped = texts.apply(lambda x: x + end_token)
    return texts_stopped


def build_vocabulary(texts, end_token='\n') -> set:
    """
    builds a set containing every unique character in the texts
    :param end_token: end token
    :param texts: pandas Series containing all texts
    :return: vocabualary set
    """
    vocab = {end_token}  # add stop token
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


def gen_input_and_target(text_inputs, char_to_idx: dict, vocab: set, seq_length: int = 20, step: int = 1,
                         pickle_filename: str = None) -> tuple:
    """
    TODO
    :param text_inputs:
    :param vocab:
    :param step:
    :param seq_length:
    :param char_to_idx:
    :param pickle_filename:
    :return: dataframe w/ columns 'inputs', 'targets' containing text sequences and their next
    """
    # merge all the text together
    text_inputs = ''.join(text_inputs.to_numpy().flatten())
    print(text_inputs)

    # instantiate lists to contain sequences and next characters
    sequences = []
    next_chars = []
    # loop over each text in texts to get subset of length seq_length and the next character
    for i in range(0, len(text_inputs) - seq_length, step):
        sequences.append(text_inputs[i:i + seq_length])
        next_chars.append(text_inputs[i + seq_length])

    # create dataframe with these lists ad columns
    ml_data = pd.DataFrame({'inputs': sequences, 'targets': next_chars})

    # create vector rep. of inputs and targets
    seqs_as_int = [[char_to_idx[c] for c in seq] for seq in sequences]
    next_char_as_int = [char_to_idx[c] for c in next_chars]

    # instanciate empty vectors to contain input and target data
    numerical_texts = np.zeros(((len(text_inputs) - seq_length), seq_length + 1, len(vocab)),
                               dtype=bool)
    numerical_next_chars = np.zeros(((len(text_inputs) - seq_length), len(vocab)), dtype=bool)

    # vectorise imput and targets
    for i, text in enumerate(seqs_as_int):
        for t, idx in enumerate(text):
            numerical_texts[i, t, idx] = 1
    for i, idx in enumerate(next_char_as_int):
        numerical_next_chars[i, idx] = 1

    # print file to pickle
    if pickle_filename:
        with open(pickle_filename, 'wb') as file:
            pickle.dump(ml_data, file)
        print(f"printed dataset to file {pickle_filename}")

    return numerical_texts, numerical_next_chars


def generate_model(vocab: set, seq_len: int, rnn_units=50, learning_rate=0.0003) -> tf.keras.Model:
    """
    TODO write doc for model generation
    :param learning_rate:
    :param vocab:
    :param seq_len:
    :param rnn_units:
    :return:
    """
    model = Sequential()
    optimizer = Adam(lr=learning_rate)

    # model.add(Embedding(input_dim=len(vocab), output_dim=20, input_length=seq_len + 1))
    model.add(GRU(rnn_units, input_shape=(seq_len + 1, len(vocab)),
                  return_sequences=True, dropout=0.15, recurrent_dropout=0.15,
                  recurrent_initializer='glorot_uniform'))
    model.add(GRU(rnn_units, return_sequences=False, dropout=0.15, recurrent_dropout=0.15,
                  recurrent_initializer='glorot_uniform'))
    model.add(Dense(len(vocab), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print(model.summary())
    return model


def fit_model(model, inputs, targets, n_epochs: int = 10, batch_size: int = 64) -> tf.keras.Model:
    # create checkpoints for callbacks
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                             save_weights_only=True)

    # fit model
    print(inputs)
    print(targets)
    model.fit(inputs, targets,
              epochs=n_epochs,
              callbacks=[checkpoint_callback],
              batch_size=batch_size,
              verbose=2)
    model.save('model.h5')
    model.reset_metrics()
    return model


def generate_text(model, n: int, max_len: int, seq_len: int, vocab: set, char_to_idx: dict, end_token: str = '\n',
                  creativity=1) -> None:
    """
    TODO: write doc for text generation
    :param seq_len:
    :param model:
    :param n:
    :param max_len:
    :param vocab:
    :param char_to_idx:
    :param end_token:
    :param creativity:
    :return:
    """

    # crativity: rescale prediction based on 'temperature'
    def scale_softmax(preds, temperatrure=creativity):
        scaled_pred = np.asarray(preds).astype('float64')
        scaled_pred = np.exp(np.log(scaled_pred) / temperatrure)
        scaled_pred = scaled_pred / np.sum(scaled_pred)
        scaled_pred = np.random.multinomial(1, scaled_pred, 1)
        return np.argmax(scaled_pred)

    # generate n texts
    for i in range(0, n):
        stop = False
        counter = 1
        text = ''

        # to contain output text, initialise by filling with start character
        output_seq = np.zeros([1, seq_len + 1, len(vocab)])
        output_seq[0, 0, char_to_idx[end_token]] = 1.
        # generate new characters until you reach end token or text reaches maximum length
        while not stop and counter < max_len + 1:
            probs = model.predict_proba(output_seq, verbose=0)[0]
            # print(probs)
            # c = idx_to_char[scale_softmax(probs, creativity)]
            c = np.random.choice(sorted(list(vocab)), replace=True, p=probs.reshape(len(vocab)))
            if c == end_token:
                stop = True
            else:
                text += c
                # output_seq[0, counter, char_to_idx[c]] = 1.
                counter += 1
        print(text)
    return None


if __name__ == '__main___':
    exit(0)
