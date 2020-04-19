from functions import *


def main():
    json_to_pickle()
    data = edit_data('data_raw.pkl', print_filename='data.pkl', perform_printouts=True)
    chatnames = get_chat_names('data.pkl')
    for chatname in chatnames['chatname']:
        print(chatname)


if __name__ == '__main__':
    main()
