import glob
import os
import json
import pandas as pd


def json_to_pickle(pickle_filename='data_raw.pkl'):
    """
    TODO: more user-friendly json file input
    Reads json files
    :param pickle_filename: name of output pickle file
    :return: raw pandas dataframe
    """
    # get files
    old_path = os.path.join(r'data\new_chat\message_*.json')
    new_path = os.path.join(r'data\old_chat\message_*.json')
    file_list = glob.glob(old_path)
    file_list.extend(glob.glob(new_path))
    print("file list: ", file_list)

    # object parser for json in to fix wrong emoji encoding that facebook uses
    def parse_obj(obj):
        for key in obj:
            if isinstance(obj[key], str):
                obj[key] = obj[key].encode('latin_1').decode('utf-8')
            elif isinstance(obj[key], list):
                obj[key] = list(map(lambda x: x if type(x) != str else x.encode('latin_1').decode('utf-8'), obj[key]))
            pass
        return obj

    # files into a dataframe
    messages = []
    senders = []
    timestamps = []
    for filepath in file_list:
        print("reading", filepath, "...")
        with open(filepath) as file:
            for message in json.load(file, object_hook=parse_obj)['messages']:
                # check if messasge contains text first
                if 'content' in message:
                    sender = message['sender_name']
                    senders.append(sender)
                    text = message['content']
                    messages.append(text)
                    timestamp = message['timestamp_ms']
                    timestamps.append(timestamp)
                    # print(f'from {filepath} {sender}: <{text}> to dataframe')
    print('converting to dataframe...')
    df = pd.DataFrame({'message': messages, 'sender': senders, 'timestamp': timestamps})

    # to pickle
    df.to_pickle(pickle_filename)
    print(f'data pickled to {pickle_filename}')

    return df


def get_chat_names(pickle_file):
    """
    Returns dataframe of all chatnames and datestamps they were changed at
    WARNING: will also return a normal message containing the words 'named the group '
    :param pickle_file: path to pickle file of message dataframe with column 'message' containing message text
    :return: dataframe with columns ['timestamp','chatname']
    """
    # load file
    df = pd.read_pickle(pickle_file)
    # get all messages with a group name change
    df = df[df['message'].str.contains('named the group')]
    # return the text after the string 'named the group '
    df['chatname'] = df['message'].apply(lambda x: x[x.find('named the group ') + 16:])
    df.drop('message', axis=1, inplace=True)

    return df


def edit_data(pickle_file, print_filename=False, perform_printouts=False):
    """
    Makes the dataframe-y by turning names into categorical and timestamp column a datetime index & fixing emoji
    :param perform_printouts: if True print outs some information about the dataframe
    :param pickle_file: path to pickle file containing dataframe
    :param print_filename: if input prints out pickle file with this name
    :return: dataframe with nice clean data. columns: ['timestamp', 'sender', 'message'] index: timestamp
    """
    df = pd.read_pickle(pickle_file)

    # to datetime
    df['timsstamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # fix errington's alt account name
    erring_alt_name = "\u00e0\u00b8\u0084\u00e0\u00b8\u0084\u00e0\u00b8\u0084\u00e0\u00b8\u00b2\u00e0\u00b9\u0088" \
                      "\u00e0\u00b8\u00b0\u00e0\u00b8\u00b0\u00e0\u00b8\u00b0\u00e0\u00b8\u00a3\u00e0\u00b9\u0087 "
    df['sender'].replace({erring_alt_name: 'Alex Errington'}, inplace=True)

    # sender as categorical
    df['sender'] = df['sender'].astype('category')

    # printouts
    if print_filename:
        df.to_pickle(print_filename)
    else:
        pass
    if perform_printouts:
        print(df.head())
        print(df.info())
        print(df.describe())
        print(f'most common message: {df.describe().iloc[2, 1]}')
        print(f'Member names: {df.sender.cat.categories}')

    return df


if __name__ == '__main__':
    json_to_pickle()
    data = edit_data('data_raw.pkl', print_filename='data.pkl', perform_printouts=True)
    chatnames = get_chat_names('data.pkl')
    for chatname in chatnames['chatname']:
        print(chatname)
