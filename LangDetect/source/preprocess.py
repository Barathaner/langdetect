import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import MeCab
import pythainlp
from konlpy.tag import Okt  # Okt ist ein Tokenizer für Koreanisch

import jieba    

# Diese Funktion bleibt unverändert
def tokenize_japanese(text):
    tokenizer = MeCab.Tagger()
    tokens = []
    node = tokenizer.parseToNode(text)
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens

# Verbesserte Tokenisierung für Chinesisch
def tokenize_chinese(text):
    # Entfernen von Sonderzeichen vor der Tokenisierung
    tokens = jieba.cut(text)
    return list(tokens)

def tokenize_thai(text):
    # Verwende pythainlp zum Tokenisieren des Thai-Textes
    tokens = pythainlp.tokenize.word_tokenize(text)
    return tokens

def tokenize_korean(text):
    # Verwende konlpy's Okt zum Tokenisieren des Koreanisch-Textes
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens


def preprocess(sentence, labels):
    corpus = []
    sentence = sentence.to_list()
    labels = labels.to_list()

    for i in range(len(sentence)):
        # Entfernen von Sonderzeichen und Umwandlung in Kleinbuchstaben
        processed_sentence = sentence[i]

        if labels[i] == 'Chinese':
            tokens = tokenize_chinese(processed_sentence)
        elif labels[i] == 'Japanese':
            tokens = tokenize_japanese(processed_sentence)
        elif labels[i] == 'Thai':
            tokens = tokenize_thai(processed_sentence)
        elif labels[i] == 'Korean':
            tokens = tokenize_korean(processed_sentence)
        else:
            # Standardverhalten für andere Sprachen
            tokens = processed_sentence.split()
        
        # Wiederzusammenfügen der tokenisierten Wörter zu einem Satz
        processed_sentence = ' '.join(tokens)
        corpus.append(processed_sentence)

    sentence = pd.Series(corpus)
    labels = pd.Series(labels)
    return sentence, labels
