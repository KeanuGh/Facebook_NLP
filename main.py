from data_cleaning_functions import *
from my_nlp import *
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import re


def main():
    # import json data into raw dataframe
    # json_to_pickle('data_raw.pkl')
    # fix dataframe to be more dataframe-y
    # cleaned_data = clean_data(input_file='data_raw.pkl', output_file='data.pkl', printouts=False)

    # get chat names
    # df = get_chat_names('data.pkl', to_file=True, to_numpy=False)
    # column = 'chatname'

    # get messages from Keanu
    df = pd.read_pickle('data.pkl')
    column = 'message'
    df = df[df['sender'] == 'Keanu Ghorbanian'].loc['2019']
    # remove non-alphanumeric characters
    df[column] = df[column].str.replace('[^a-zA-Z\s]', '')

    # plot message lengths
    df['text_length'] = df[column].map(len)
    # df['text_length'].plot(kind='hist', bins=20)

    # add start and end tokens to input and output
    start_token = '\t'
    end_token = '\n'
    sequence_length = 15  # length of training sequences
    max_gen_len = 120  # maximum length of generated texts
    df['inputs'] = end_tokens(df[column], end_token=end_token)

    # build vocabulary
    vocab = build_vocabulary(df[column], end_token=end_token)
    print(f'number of text samples: {len(df)}')
    print(f'vocabulary list: {sorted(list(vocab))}')
    print(f'vocabulary size: {len(vocab)}')

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # build character-to-index dictionaries for vectorisation
    char_to_idx, idx_to_char = char_to_int_maps(vocab)
    inputs, targets = gen_input_and_target(df['inputs'],
                                           vocab=vocab,
                                           char_to_idx=char_to_idx,
                                           seq_length=sequence_length,
                                           vectorize=True,
                                           print_test=True,
                                           pickle_filename='tf_dataset.pkl')

    # generate model
    # model = generate_model(vocab, rnn_units=1024, seq_len=sequence_length, embedding=False)
    # model = fit_model(model, inputs, targets, n_epochs=10)
    model = load_model('model.h5')
    test_string = [char_to_idx[c] for c in 'test string']
    test_string = [idx_to_char[i] for i in test_string]
    print(''.join(test_string))
    generate_text(model, n=10, max_len=max_gen_len, seq_len=sequence_length, vocab=vocab,
                  char_to_idx=char_to_idx, idx_to_char=idx_to_char, embedding=False)


def quick_generate(n_gen=10, vocab_file='vocab.pkl'):
    vocab = pickle.load(open(vocab_file, 'r', encoding="utf-8"))
    char_to_idx, idx_to_char = char_to_int_maps(vocab)
    model = load_model('model.h5')
    generate_text(model, n=n_gen, max_len=120, seq_len=20, vocab=vocab,
                  char_to_idx=char_to_idx)


if __name__ == '__main__':
    main()
    # quick_generate()
