from data_cleaning_functions import *
import matplotlib.pyplot as plt
import pandas as pd

# shut the hell up pandas i know what I'm doing (narrator voice: he did not know what he was doing)
pd.options.mode.chained_assignment = None


def check_message_type(message_type: str):
    """
    Check if message type is a valid message type
    :param message_type: input type
    :return: True if is valid. If not raises exception and gives helpful error message
    """
    if type(message_type) != str:
        raise Exception("Message type must be formatted as a string")

    valid_types = {'PHOTO', 'VIDEO', 'TEXT', 'FILE', 'STICKER', 'GIF', 'AUDIO'}
    if message_type.upper() not in valid_types:
        raise Exception(f"Invalid message type. Valid types: {valid_types} (case insensitive)")
    else:
        return True


def whos_said_x_most(dataframe: pd.DataFrame, x: str):
    dataframe = dataframe[dataframe.message.str.contains(rf'\b{x}\b', na=False, case=False)]
    print(f"the word '{x}' has been said {len(dataframe)} times!")
    print(dataframe.sender.value_counts())
    return dataframe


def n_most_reacted_messages(dataframe: pd.DataFrame, message_type=None):
    dataframe = dataframe[dataframe['reactions'].notna()]
    dataframe['n_reacts'] = dataframe.reactions.apply(lambda x: len(x))

    check_message_type(message_type)
    dataframe = dataframe[dataframe['category'] == message_type.upper()]

    dataframe.sort_values(by='n_reacts', ascending=False, inplace=True)
    return dataframe


def most_posters(dataframe: pd.DataFrame, plot: bool = False, title=''):
    posters = dataframe.sender.value_counts().dropna()

    if plot:
        posters.plot(kind='bar')
        plt.title(title)
        plt.show()
    posters.sort_values(inplace=True, ascending=True)
    return posters


def gen_wordcloud(dataframe: pd.DataFrame, stopwords=None):
    """
    generates a wordcloud
    :param dataframe: input dataframe
    :param stopwords: set of user stopwords. If you want to include all words use stopwords = 'off'
    :return: None
    """
    import nltk
    from wordcloud.wordcloud import WordCloud

    # get stopwords from nltk and merge with given stopwords
    nltk.download('stopwords')
    if stopwords == 'off':
        stopwords = {}
    elif stopwords is not None:
        stopwords = stopwords | set(nltk.corpus.stopwords.words('english'))
    else:
        stopwords = nltk.corpus.stopwords.words('english')

    dataframe = dataframe[dataframe['category'] == 'TEXT']
    text = dataframe['message'].str.cat()
    cloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def reacts_per_message(dataframe: pd.DataFrame, printout: bool = False, message_type=None):
    """
    Calculates the number of reactions a poster has gotten divided by the number of times posted
    :param message_type: string or tuple of strings
    :param printout:
    :param dataframe:
    :return:
    """

    if type(message_type) == str:
        check_message_type(message_type)
        dataframe = dataframe.loc[dataframe['category'] == message_type.upper()]
    elif type(message_type) == tuple:
        _ = [check_message_type(m_type) for m_type in message_type]
        dataframe = dataframe[dataframe.category.apply(lambda x: any(typ for typ in message_type if typ.upper() in x))]
    else:
        raise Exception("Please format message_type as a string or tuple of strings")

    dataframe['n_reacts'] = dataframe['reactions'].apply(lambda x: len(x) if x is not None else x)

    react_counts = dataframe.groupby('sender')['n_reacts'].count()
    message_counts = dataframe.sender.value_counts()

    react_rate = react_counts.divide(message_counts).dropna()
    react_rate.sort_values(inplace=True, ascending=False)
    if printout:
        print(react_rate)
    return react_rate


def who_reacts_the_most(dataframe: pd.DataFrame, plot: bool = False, printout: bool = False, title: str = ''):
    # extract names from reactions
    reactions = dataframe['reactions'].dropna().to_numpy()
    names = [name for emoji, name in [react for reaction in reactions for react in reaction]]

    # count unique values
    names, counts = np.unique(names, return_counts=True)

    # sort by most reacts
    count_sort_ind = np.argsort(-counts)
    names = names[count_sort_ind]
    counts = counts[count_sort_ind]

    # plot
    if plot:
        plt.bar(names, counts)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.show()

    # combine
    n_reacts = list(zip(names, counts))

    # printout
    if printout:
        print(n_reacts)

    return n_reacts


def clean_emoji(dataframe: pd.DataFrame, column: str):
    """
    cleans non-standard emoji from dataframe column 'column' and maps heart reacts to heart
    """
    # combine heart reacts
    dataframe[column].replace({'💗': '❤', '😍': '❤'}, inplace=True)

    # restrict reacts to only the basic ones
    allowed_reacts = {'❤', '👍', '👎', '😢', '😠', '😆', '😮'}
    dataframe = dataframe[dataframe[column].isin(allowed_reacts)]

    return dataframe


def who_uses_which_react(dataframe: pd.DataFrame, plot: bool = False, vmax=None, as_fraction: bool = False):
    """
    Who uses which react the most, only takes into account the 'basic' facebook reacts. Casts all 'heart' reacts to '❤'
    :param as_fraction:
    :param dataframe: input dataframe
    :param plot: whether to plot a heatmap
    :param vmax: cap for heatmap in case of outliers
    :return: dataframe with colmns corresponding to emoji,
    """

    # split reactions into separate rows
    dataframe = dataframe['reactions'].dropna().explode('reactions')

    # get dataframe of only emojis and the name of the person who sent that react
    dataframe = pd.DataFrame(dataframe.tolist(), index=dataframe.index)
    dataframe.columns = ['emoji', 'name']

    # create count column containing counts for emoji, name combinations
    dataframe = dataframe.groupby(['emoji', 'name']).size().reset_index().rename(columns={0: 'count'})

    dataframe = clean_emoji(dataframe, 'emoji')

    # PIVOT
    dataframe = dataframe.pivot_table(index='name', columns='emoji', values='count', fill_value=0)

    # have all counts be as a fraction of total reactions for user
    if as_fraction:
        dataframe['sum'] = dataframe.sum(axis=1)
        dataframe = dataframe.div(dataframe["sum"], axis=0).drop(['sum'], axis=1)

    # plot heatmap
    if plot:
        # to get emoji to show up properly
        import matplotlib
        matplotlib.rcParams.update({'font.family': 'Segoe UI Emoji'})

        # plot
        from seaborn import heatmap
        if as_fraction:
            fmt = '.0%'
            title = "percentage-wise counts of users sending each react"
        else:
            fmt = 'd'
            title = "Counts of users sending each react"
        heatmap(dataframe,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                cbar=False,
                xticklabels=True,
                yticklabels=True,
                )
        plt.title(title)
        plt.show()

    return dataframe


def who_recieves_which_react(dataframe: pd.DataFrame, plot: bool = False, vmax=None, as_fraction: bool = False):
    """
    Who recieves which react for messages they send?
    :return:
    """
    # get just names and the reactions for messages with reacts
    dataframe = dataframe[dataframe['reactions'].notna()].reset_index()[['sender', 'reactions']]

    # make react column a list of each react the message received (remove who reacted)
    def reduce_to_emoji(react_tuple):
        return [emoji for emoji, name in react_tuple]
    dataframe['reactions'] = dataframe['reactions'].apply(reduce_to_emoji)
    dataframe.columns = ['name', 'emoji']

    # expand to separate rows
    dataframe = dataframe.explode('emoji')

    # clean emoji
    dataframe = clean_emoji(dataframe, 'emoji')
    print(dataframe)

    # create count column containing counts for emoji, name combinations
    dataframe = dataframe.groupby(['emoji', 'name']).size().reset_index().rename(columns={0: 'count'})

    # PIVOT
    dataframe = dataframe.pivot_table(index='name', columns='emoji', values='count', fill_value=0)

    # delete zero rows
    dataframe = dataframe[(dataframe.T != 0).any()]

    # have all counts be as a fraction of total reactions for user
    if as_fraction:
        dataframe['sum'] = dataframe.sum(axis=1)
        dataframe = dataframe.div(dataframe["sum"], axis=0).drop(['sum'], axis=1)

    # plot heatmap
    if plot:
        # to get emoji to show up properly
        import matplotlib
        matplotlib.rcParams.update({'font.family': 'Segoe UI Emoji'})

        # plot
        from seaborn import heatmap
        if as_fraction:
            fmt = '.0%'
            title = "percentage-wise counts of users recieving each react"
        else:
            fmt = 'd'
            title = "Counts of users recieving each react"
        heatmap(dataframe,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                cbar=False,
                xticklabels=True,
                yticklabels=True,
                )
        plt.title(title)
        plt.show()

    return dataframe


if __name__ == '__main__':
    datafile = '2020-09-01_data.pkl'
    datafile_clean = '2020-09-01_data_CLEAN.pkl'

    # # extract and clean data
    # json_to_pickle(datafile)
    # clean_data(datafile, datafile_clean, printouts=False)

    df = pd.read_pickle(datafile_clean)
    df_2020 = df[df.index.year == 2020]

    # x_messages = whos_said_x_most(df, 'Media')
    #
    # data = n_most_reacted_messages(df, message_type='TEXT')
    # print(data['message'][0])

    # stop = {'kéo', 'watch', 'youtube', 'www', 'com'}
    # gen_wordcloud(df, stopwords=stop)

    # most_posters(df_2020, plot=True, title="number of messages sent in 2020")

    # most_reacted = who_reacts_the_most(df_2020, printout=True, plot=True,
    # title="Number of times reacting to a post in 2020")

    # reaction_rates = reacts_per_message(df_2020, message_type=None)
    # reaction_rates.plot(kind='bar', title='average number of reactions per any message type in 2020')
    # plt.show()

    # data = who_uses_which_react(df_2020, plot=True, vmax=2500)

    data = who_recieves_which_react(df, plot=True, as_fraction=True)
    print(data.head())
