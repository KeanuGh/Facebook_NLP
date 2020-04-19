from data_cleaning_functions import *
from nlp import *
import os


def import_and_clean_files():
    # import json data into raw dataframe
    json_to_pickle('data_raw.pkl')
    # fix dataframe to be more dataframe-y
    data = edit_data(input_file='data_raw.pkl', output_file='data.pkl', perform_printouts=True)

    # get chat names
    chatnames_df = get_chat_names('data.pkl', to_file=True)

    # check lengths of chatnames
    chatnames_df['text_length'] = [len(message) for message in chatnames_df['chatname']]
    # chatnames_df['text_length'].plot(kind='hist', bins=30)
    # filter out group names with length > 120
    chatnames_df = chatnames_df[chatnames_df['text_length'] < 120]

    # add start and end tokens to input and output
    chatnames_df['chatnames_processed'] = start_and_end_tokens(chatnames_df['chatname'])

    # build vocabulary
    vocab = build_vocabulary(chatnames_df['chatname'])
    print(f'vocabulary list: {sorted(list(vocab))}')
    print(f'vocabulary size: {len(vocab)}')

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # build character-to-index dictionaries for vectorisation
    char_to_idx, idx_to_char = char_to_int_maps(vocab)

    # get tensorflow dataset of inputs and targets
    dataset = gen_input_and_target(chatnames_df, 'chatnames_processed', 20, char_to_idx,
                                   pickle_filename='tf_dataset.pkl')


def create_and_train_model():
    vocab = pickle.load(open('vocab.pkl', 'rb'))
    model = generate_model(vocab, rnn_units=100)


if __name__ == '__main__':
    # disables AVX/FMA warning as I'm using GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # imports data, cleans and processes to tensorflow dataset for model input
    import_and_clean_files()

    # creates and trains model from tensorflow dataset pickle file
    create_and_train_model()
