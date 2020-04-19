from functions import *
from nlp import *


def main():
    chatnames_df = get_chat_names('data.pkl', to_file=True)
    chatnames_df['stopped_text'] = start_and_end_tokens(chatnames_df['chatname'])
    vocab = build_vocabulary(chatnames_df['stopped_text'])
    print(sorted(list(vocab)))
    print(len(vocab))


if __name__ == '__main__':
    main()
