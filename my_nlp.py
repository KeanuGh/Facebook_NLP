import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


def end_tokens(texts: np.array, end_token: str = '\n') -> tuple:
    """
    adds token (default '\n') to the end of each text in an array
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
                         pickle_filename: str = None, vectorize=True, print_test=False) -> tuple:
    """
    Generates input and target vectors
    :param vectorize: if True returns vectorised sequences. declare False for input into an embedding layer
    :param text_inputs: pandas series of input texts
    :param vocab: vocabulary set
    :param step: step for sequence generation. increasing this number will decreasse the number of training examples
    :param seq_length: length of sequence chunk
    :param char_to_idx: character-to-index mapping dictionary
    :param pickle_filename: name of file to pickle output to
    :param print_test: print out 5 training examples: sequence and next char
    :return: dataframe w/ columns 'inputs', 'targets' containing text sequences and their next character
    """
    # merge all the text together
    text_inputs = ''.join(text_inputs.to_numpy().flatten())

    # instantiate lists to contain sequences and next characters
    sequences = []
    targets = []
    # loop over each text in texts to get subset of length seq_length and the next character
    if vectorize:  # for no embedding layer, target is the next character
        for i in range(0, len(text_inputs) - seq_length, step):
            sequences.append(text_inputs[i:i + seq_length])
            targets.append(text_inputs[i + seq_length])
    else:  # for an embedding layer, target is the next sequence
        for i in range(0, len(text_inputs) - seq_length, step):
            chunk = text_inputs[i:i + seq_length + 1]
            sequences.append(chunk[:-1])
            targets.append(chunk[1:])

    # print file to pickle
    if pickle_filename:
        # create dataframe with these lists ad columns
        ml_data = pd.DataFrame({'inputs': sequences, 'targets': targets})
        with open(pickle_filename, 'wb') as file:
            pickle.dump(ml_data, file)
        print(f"printed dataset to file {pickle_filename}")

    # print training examples: a sequence and next character
    if print_test:
        print(f"number of input sequences: {len(sequences)}")
        idxs = np.random.choice(len(sequences), size=5, replace=False)
        for idx in idxs:
            print(f'\nInput: {sequences[idx]}\nTarget: {targets[idx]}')

    # create vector rep. of inputs and targets
    seqs_as_int = [[char_to_idx[c] for c in seq] for seq in sequences]
    if vectorize:
        target_as_int = [char_to_idx[c] for c in targets]
    else:
        target_as_int = [[char_to_idx[c] for c in seq] for seq in targets]

    # if vectorise is true, vectorise input and target
    # creates 3d vector for inputs (sequences) and 2d vector for target (next characters)
    if vectorize:
        # instanciate empty vectors to contain input and target data
        numerical_sequences = np.zeros(((len(text_inputs) - seq_length), seq_length + 1, len(vocab)),
                                       dtype=bool)
        numerical_targets = np.zeros(((len(text_inputs) - seq_length), len(vocab)), dtype=bool)

        # vectorise imput and targets
        for i, text in enumerate(seqs_as_int):
            for t, idx in enumerate(text):
                numerical_sequences[i, t, idx] = 1
        for i, idx in enumerate(target_as_int):
            numerical_targets[i, idx] = 1
        return numerical_sequences, numerical_targets
    else:
        # from keras.utils import to_categorical
        # target_as_int = to_categorical(target_as_int)
        return seqs_as_int, target_as_int


def generate_model(vocab: set, seq_len: int, rnn_units=50, learning_rate=0.001, embedding=False,
                   embedding_dim=256) -> tf.keras.Model:
    """
    TODO write doc for model generation
    :param embedding_dim:
    :param embedding:
    :param learning_rate:
    :param vocab:
    :param seq_len:
    :param rnn_units:
    :return:
    """
    model = Sequential()
    optimizer = Adam(lr=learning_rate)
    if embedding:  # if embedding layer need to add first LSTM layer without input shape declared
        model.add(Embedding(len(vocab), embedding_dim,
                            trainable=True, embeddings_initializer=None))
        model.add(LSTM(rnn_units, return_sequences=True, dropout=0.15, recurrent_dropout=0.15,
                       recurrent_initializer='glorot_uniform'))
    else:  # if no embedding layer need to declare inputs in first layer
        model.add(LSTM(rnn_units, input_shape=(seq_len + 1, len(vocab)),
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
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    # fit model
    inputs = np.array(inputs)
    targets = np.array(targets)
    print(f'input: {inputs.shape}, {targets.dtype}\ntarget: {targets.shape}, {targets.dtype}')
    model.fit(inputs, targets, epochs=n_epochs, callbacks=[checkpoint_callback], batch_size=batch_size,
              validation_split=0.2, verbose=2)
    model.save('model.h5')
    model.reset_metrics()
    return model


def generate_text(model, n: int, max_len: int, seq_len: int, vocab: set, char_to_idx: dict,
                  idx_to_char: dict, end_token: str = '\n', creativity=1, embedding=False) -> None:
    """
    TODO: write doc for text generation
    :param idx_to_char:
    :param embedding:
    :param seq_len: length of input sequences
    :param model: tf model for next character prediction
    :param n: number of texts to generate
    :param max_len: maximum length of generated texts
    :param vocab: vocebulary set
    :param char_to_idx: character to index encoding dict
    :param end_token: ending token
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
    for i in range(n):
        stop = False
        text = ''

        if embedding:
            output_seq = [char_to_idx['\n']]
            input_eval = tf.expand_dims(output_seq, 0)
            while not stop and len(text) < max_len + 1:
                predictions = model(input_eval)
                predictions = predictions / creativity
                predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
                input_eval = tf.expand_dims([predicted_id], 0)
                if idx_to_char[predicted_id] == end_token:
                    stop = True
                else:
                    input_eval = tf.expand_dims([predicted_id], 0)
                    text += idx_to_char[predicted_id]
            print(text)

        else:
            # to contain output text, initialise by filling with start character
            output_seq = np.zeros([1, seq_len + 1, len(vocab)])
            for j, c in enumerate('I never '):  # input sequence
                output_seq[0, j, char_to_idx[c]] = 1.
            # generate new characters until you reach end token or text reaches maximum length
            while not stop and len(text) < max_len + 1:
                probs = model.predict_proba(output_seq, verbose=0)[0]
                # print(probs)
                c = idx_to_char[scale_softmax(probs, creativity)]
                # c = np.random.choice(sorted(list(vocab)), replace=True, p=probs.reshape(len(vocab)))
                if c == end_token:
                    stop = True
                else:
                    text += c
                    # shift output sequence
                    output_seq[0] = np.vstack((output_seq[0, 1:], np.zeros(len(vocab))))
                    output_seq[0, -1, char_to_idx[c]] = 1

            print(text)
    return None


if __name__ == '__main___':
    print("don't run this file you idiot")
