import glob
import os
import json
import pandas as pd
import numpy as np


def json_to_pickle(pickle_filename: str = 'data_raw.pkl'):
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

    # messages into a dataframe
    messages = []
    senders = []
    timestamps = []
    reactions = []
    category = []

    # content cases with multiple possible values in one message
    dict_cases = {
        'photos': 'PHOTO',
        'videos': 'VIDEO',
        'gifs': 'GIF',
        'audio_files': 'AUDIO',
        'files': 'FILE',
    }

    for filepath in file_list:
        # print(f"reading file {filepath}...")
        with open(filepath) as file:
            for message in json.load(file, object_hook=parse_obj)['messages']:

                # any weird message types?
                if message['type'] not in ['Generic', 'Share', 'Subscribe', 'Unsubscribe', 'Call']:
                    print(f"Weird message type: {message['type']}")

                # if text message
                if 'content' in message:
                    # set category TEXT
                    category.append('TEXT')
                    # message content
                    content = message['content']
                    messages.append(content)
                # if media
                elif [i for i in dict_cases if i in message]:
                    # get the single element of the set dict_cases within the message
                    case = set(message.keys()).intersection(dict_cases).pop()
                    # set category
                    category.append(dict_cases[case])
                    # content
                    contents = []
                    for content in message[case]:
                        contents.append(content['uri'])
                    messages.append(contents)
                # if sticker
                elif 'sticker' in message:
                    # set category
                    category.append('STICKER')
                    # message content
                    content = message['sticker']['uri']
                    messages.append(content)
                # if deleted message, message field is None
                else:
                    messages.append(None)
                    category.append('DELETED')

                # name
                sender = message['sender_name']
                senders.append(sender)

                # timestamp
                timestamp = message['timestamp_ms']
                timestamps.append(timestamp)

                # reactions
                # appends list of tuples (reaction, reactor)
                if 'reactions' in message:
                    message_reactions = []
                    for reaction in message['reactions']:
                        emoji = reaction['reaction']
                        reactor = reaction['actor']
                        message_reactions.append((emoji, reactor))
                    reactions.append(message_reactions)
                else:
                    reactions.append(None)

    print('converting to dataframe...')
    df = pd.DataFrame({
        'message': messages,
        'sender': senders,
        'timestamp': timestamps,
        'reactions': reactions,
        'category': category,
        }
    )

    # to pickle
    df.to_pickle(pickle_filename)
    print(f'data pickled to {pickle_filename}')
    print(df.head())

    return df


def clean_data(input_file: str, output_file: str = None, printouts: bool = False):
    """
    Makes the data dataframe-y by turning names into categorical and timestamp column a datetime index & fixing emoji
    :param printouts: if True print outs some information about the dataframe
    :param input_file: path to pickle file containing dataframe
    :param output_file: if input prints out pickle file with this name
    :return: dataframe with nice clean data. columns: ['timestamp', 'sender', 'message'] index: timestamp
    """
    df = pd.read_pickle(input_file)

    # to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # fix errington's alt account name
    erring_alt_name = "คคค า่ะะะร็"
    df['sender'].replace({erring_alt_name: 'Alex Errington'}, inplace=True)

    # sender & category as categorical
    df['sender'] = df['sender'].astype('category')
    df['category'] = df['category'].astype('category')

    # replace None values with nan values
    # df.fillna(value=np.nan, inplace=True)

    # printouts
    if output_file is not None:
        df.to_pickle(output_file)

    if printouts:
        print(df.head())
        print(df.info())
        print(df.describe())
        # top_message = df.describe()['message']['top']
        # print(f'most common message: {top_message}')
        # print(f'Member names: {df.sender.cat.categories}')

    return df


def get_chat_names(data_file, to_file=False, to_numpy=False):
    """
    Returns dataframe of all chatnames and datestamps they were changed at
    WARNING: will also return a normal message containing the words 'named the group '
    :param to_numpy: whether to return as a numpy array
    :param to_file: if True will create pickle file of chatnames & their timestamps
    :param data_file: path to pickle file of message dataframe with column 'message' containing message text
    :return: dataframe with columns ['timestamp','chatname']
    """
    # load file
    if type(data_file) is str:
        df = pd.read_pickle(data_file)
    else:
        df = data_file

    # get all messages with a group name change
    df = df[df['message'].str.contains('named the group')]

    # return the text after the string 'named the group '
    df['chatname'] = df['message'].apply(lambda x: x[x.find('named the group ') + 16:])
    df.drop('message', axis=1, inplace=True)

    # drop chatnames containing a newline character because that's just crazy
    df = df[~df.chatname.str.contains('\n')]

    # to file(s)
    if to_file:
        df.to_pickle('chatnames.pkl')
        with open('chatnames.txt', 'w', encoding="utf-8") as file:
            for i, row in df.iterrows():
                file.write(i.strftime("%Y-%m-%d %H:%M ") + row.sender + ': ' + row.chatname + '\n')

    if to_numpy:
        return df.chatname.to_numpy()
    else:
        return df


def pkl_to_txt(input_file: str, output_file: str = 'data.txt') -> None:
    """
    prints data from .pkl file to .txt file
    """
    df = pd.read_pickle(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for t, msg, sender in df.itertuples():
            msg.replace('\n', ' ')
            f.write(t.strftime("%m/%d/%Y %H:%M:%S ") + sender + ": " + msg + '\n')
    print(f"written to file {output_file}")


def pkl_to_txt_chatnames(input_file: str = 'chatnames.pkl',
                         output_file: str = 'chatnames.txt') -> None:
    """
    prints data from .pkl file to .txt file (specifically for chat names)
    """
    df = pd.read_pickle(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for t, sndr, name in df.itertuples():
            f.write(t.strftime("%m/%d/%Y %H:%M:%S ") + ": " + name + '\n')
    print(f"written to file {output_file}")


if __name__ == '__main__':
    pkl_to_txt_chatnames()
