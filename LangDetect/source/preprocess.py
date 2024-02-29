import pandas as pd
import nltk
import MeCab
import pythainlp
from konlpy.tag import Okt  # Okt is a tokenizer for Korean.
import jieba
import random
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
    

    chinesecount = 0
    japanesecount = 0
    thaicount = 0
    koreancount = 0
    othercount = 0
    for i in range(len(sentence)):
        processed_sentence = sentence[i]
        chinesecount = 0
        japanesecount = 0
        thaicount = 0
        koreancount = 0
        othercount = 0
        #0.2 is the percentage of the sentence that will be randomly selected to determine the language
        randcount = int(len(processed_sentence) *0.2)
        randchars = random.sample(range(0, len(processed_sentence)), randcount)
        for randchar in randchars:
            #chinese Basic CJK Unified Ideographs Block and CJK Unified Ideographs Extension A Block
            if 19968 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 40959  or 13312 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 19903:
                chinesecount += 1
                #Hiragana and Katakana
            elif 12352 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 12447 or 12448 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 12543 :
                japanesecount += 1
                #thai
            elif 3584 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 3711:
                thaicount += 1
                #Hangul Syllables: 44032-55215 Hangul Jamo: 4352-4607 Hangul Compatibility Jamo: 43360-43391 Hangul Jamo Extended-B:  55216 bis 55295
            elif 44032 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 55215 or 4352 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 4607 or 43360 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 43391 or 55216 < ord(processed_sentence[randchar]) and ord(processed_sentence[randchar]) < 55295:
                koreancount += 1
            else:
                othercount += 1
        
        if max(thaicount, japanesecount, chinesecount, koreancount,othercount) == thaicount:
            tokens = tokenize_thai(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == japanesecount:
            tokens = tokenize_japanese(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == chinesecount:
            tokens = tokenize_chinese(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == koreancount:
            tokens = tokenize_korean(processed_sentence)
        else:
            #other languages are quiet similar to english so we can use the nltk tokenizer
            tokens = nltk.word_tokenize(processed_sentence)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        # this is stupid but it works. normally we would need to add new labels for each word in the sentence but we are not doing that here
        # Keep in mind that sentence splitting affectes the number of sentences
        # and therefore, you should replicate labels to match. like this.... but for some strange reason it performs really bad.... (maybe the special chcaracters)
        #   for k in range(len(tokens)):
        #    corpus.append(tokens[k])
        #    labelcorpus.append(labels[i])
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        
        processed_sentence = ' '.join(tokens)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

        corpus.append(processed_sentence)

    sentence = pd.Series(corpus)
    labels = pd.Series(labels)
    return sentence, labels
