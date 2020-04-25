from data_cleaning_functions import *
from nlp import *
from tensorflow.keras.models import load_model
import pandas as pd
import pickle


def main():
    # import json data into raw dataframe
    # json_to_pickle('data_raw.pkl')
    # fix dataframe to be more dataframe-y
    # cleaned_data = clean_data(input_file='data_raw.pkl', output_file='data.pkl', printouts=False)

    # get chat names
    chatnames_df = get_chat_names('data.pkl', to_file=True, to_numpy=False)

    # create column containing lengths of chatnames
    chatnames_df['text_length'] = chatnames_df['chatname'].map(len)
    # chatnames_df['text_length'].plot(kind='hist', bins=20)
    # filter out group names with length > 120
    # [chatnames_df = chatnames_df[chatnames_df['text_leng]th'] < 120]

    # add start and end tokens to input and output
    start_token = '\t'
    end_token = '\n'
    chatnames_df['inputs'] = start_and_end_tokens(chatnames_df['chatname'], end_token=end_token)

    # build vocabulary
    vocab = build_vocabulary(chatnames_df['chatname'], end_token=end_token)
    print(f'vocabulary list: {sorted(list(vocab))}')
    print(f'vocabulary size: {len(vocab)}')

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # build character-to-index dictionaries for vectorisation
    char_to_idx, idx_to_char = char_to_int_maps(vocab)
    inputs, targets = gen_input_and_target(chatnames_df['inputs'],
                                           vocab=vocab,
                                           char_to_idx=char_to_idx,
                                           seq_length=20,
                                           pickle_filename='tf_dataset.pkl')

    # generate model
    model = generate_model(vocab, rnn_units=128, seq_len=20)
    model = fit_model(model, inputs, targets, n_epochs=5)
    # model = load_model('model.h5')
    generate_text(model, n=10, max_len=120, seq_len=20, vocab=vocab,
                  char_to_idx=char_to_idx)


def quick_generate(n_gen=10, vocab_file='vocab.pkl'):
    vocab = pickle.load(open(vocab_file, 'r', encoding="utf-8"))
    char_to_idx, idx_to_char = char_to_int_maps(vocab)
    model = load_model('model.h5')
    generate_text(model, n=n_gen, max_len=120, seq_len=20, vocab=vocab,
                  char_to_idx=char_to_idx)


if __name__ == '__main__':
    main()
    # quick_generate()
