import pandas as pd
import nltk
import MeCab
import pythainlp
from konlpy.tag import Okt  # Okt is a tokenizer for Korean.
import jieba
from cltk import NLP
cltk_nlp = NLP(language="lat")
nltk.download('punkt')
def tokenize_japanese(text):
    """
    Tokenizes Japanese text using MeCab.

    Parameters:
    - text (str): The Japanese text to be tokenized.

    Returns:
    - list: A list of tokens extracted from the input text.
    """
    tokenizer = MeCab.Tagger()
    tokens = []
    node = tokenizer.parseToNode(text)
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens

def tokenize_chinese(text):
    """
    Tokenizes Chinese text using Jieba.

    Parameters:
    - text (str): The Chinese text to be tokenized.

    Returns:
    - list: A list of tokens extracted from the input text.
    """
    tokens = jieba.cut(text)
    return list(tokens)


def tokenize_thai(text):
    """
    Tokenizes Thai text using PyThaiNLP.

    Parameters:
    - text (str): The Thai text to be tokenized.

    Returns:
    - list: A list of tokens extracted from the input text.
    """
    tokens = pythainlp.tokenize.word_tokenize(text)
    return tokens

def tokenize_korean(text):
    """
    Tokenizes Korean text using Okt from konlpy.

    Parameters:
    - text (str): The Korean text to be tokenized.

    Returns:
    - list: A list of tokens extracted from the input text.
    """
    okt = Okt()
    tokens = okt.morphs(text)
    return tokens


def preprocess(sentence, labels):
    """
    Preprocesses text data for various languages by applying language-specific tokenization.

    Parameters:
    - sentence (pd.Series): A pandas Series containing sentences to be preprocessed.
    - labels (pd.Series): A pandas Series containing labels indicating the language of each sentence.

    Returns:
    - tuple: A tuple containing two pandas Series: the preprocessed sentences and the original labels.

    The function supports preprocessing for Chinese, Japanese, Thai, and Korean texts. For languages not specifically supported, it defaults to simple whitespace-based tokenization. This preprocessing step is crucial for natural language processing tasks, especially when dealing with languages that do not use whitespace to separate words.
    """
    corpus = []
    sentence = sentence.to_list()
    labels = labels.to_list()
    

    for i in range(len(sentence)):
        processed_sentence = sentence[i]

        if labels[i] == 'Chinese':
            tokens = tokenize_chinese(processed_sentence)
        elif labels[i] == 'Japanese':
            tokens = tokenize_japanese(processed_sentence)
        elif labels[i] == 'Thai':
            tokens = tokenize_thai(processed_sentence)
        elif labels[i] == 'Korean':
            tokens = tokenize_korean(processed_sentence)
        elif labels[i] == 'Tamil':
            tokens=processed_sentence.split()
        elif labels[i] == 'Urdu':
            tokens=processed_sentence.split()
        elif labels[i] == 'Persian':
            tokens=processed_sentence.split()
        elif labels[i] == 'Pushto':
            tokens=processed_sentence.split()
        elif labels[i] == 'Romanian':
            tokens=processed_sentence.split()
        elif labels[i] == 'Arabic':
            tokens=processed_sentence.split()
        elif labels[i] == 'Hindi':
            tokens=processed_sentence.split()
        elif labels[i] == 'Latin':
            cltk_doc = cltk_nlp.analyze(text=processed_sentence)
            tokens=cltk_doc.tokens
        elif labels[i] == 'Indonesian':
            tokens=processed_sentence.split()
        else:
            lang = labels[i].lower()
            if lang == 'portugese':
                lang = 'portuguese'
            tokens = nltk.tokenize.word_tokenize(text=processed_sentence,language=lang)
        
        processed_sentence = ' '.join(tokens)
        corpus.append(processed_sentence)

    sentence = pd.Series(corpus)
    labels = pd.Series(labels)
    return sentence, labels
