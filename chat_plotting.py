from data_cleaning_functions import *
import matplotlib.pyplot as plt
from wordcloud.wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords


def whos_said_x_most(x):
    data = df[df.message.str.contains(rf'\b{x}\b', na=False, case=False)]
    print(f"the word '{x}' has been said {len(data)} times!")
    print(data.sender.value_counts())
    return data


def n_most_reacted_messages(n=10, message_type=None):
    data = df[df['reactions'].notna()]
    data['n_reacts'] = data['reactions'].apply(lambda x: len(x))

    if message_type in ['PHOTO', 'VIDEO', 'TEXT', 'FILE', 'STICKER', 'GIF']:
        data = data[data['type'] == message_type]
    elif message_type is not None:
        raise Exception(f"No message type {message_type}")

    data.sort_values(by='n_reacts', ascending=False, inplace=True)
    print(data.head(n))
    return data


def most_posters(dataframe, title):
    posters = dataframe.sender.value_counts()
    posters.plot(kind='bar')
    plt.title(title)
    plt.show()


def gen_wordcloud(dataframe, stopwords=None):
    dataframe = dataframe[dataframe['category'] == 'TEXT']
    text = dataframe['message'].str.cat()
    cloud = WordCloud(stopwords=stopwords).generate(text)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    datafile = '2020-09-01_data.pkl'
    datafile_clean = '2020-09-01_data_CLEAN.pkl'

    # # extract and clean data
    # json_to_pickle(datafile)
    # clean_data(datafile, datafile_clean, printouts=False)

    df = pd.read_pickle(datafile_clean)
    df_2020 = df[df.index.year == 2020]

    # x_messages = whos_said_x_most('kéo')
    #
    # data = n_most_reacted_messages()
    # print(data['message'][0])

    #
    # nltk.download('stopwords')
    # stopwords = {'kéo', 'watch', 'youtube', 'www', 'com'} | set(stopwords.words('english'))
    # gen_wordcloud(df, stopwords=stopwords)

    # most_posters(df_2020, "messages 2020")


