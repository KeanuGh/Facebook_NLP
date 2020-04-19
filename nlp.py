import pandas as pd


def start_and_end_tokens(texts, start_token=r'\t', end_token=r'\n'):
    new_texts = texts.apply(lambda x: r'\\\t' + x + r'\\\n')
    return new_texts


def build_vocabulary(texts):
    vocab = {r'\\\t', r'\\\n'}
    for text in texts:
        for c in text:
            if c not in vocab:
                vocab.add(c)
    return vocab


