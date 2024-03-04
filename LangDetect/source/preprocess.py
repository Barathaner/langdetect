import pandas as pd
import nltk
import MeCab
import pythainlp
from konlpy.tag import Okt  # Okt is a tokenizer for Korean.
import jieba
import random
from nltk.stem import SnowballStemmer
import re
nltk.download('punkt')



def stem_text(text, stemmer):
    stemmed_words = [stemmer.stem(word) for word in text]
    return stemmed_words

def remove_special_characters(sentence):
    # Extended function that checks if a character is a special character
    def is_special_character(char):
        codepoint = ord(char)
        # Extended selection of Unicode blocks for special characters, including some punctuation marks and symbols
        if (0x20A0 <= codepoint <= 0x20CF) or \
           (0x2200 <= codepoint <= 0x22FF) or \
           (0x2300 <= codepoint <= 0x23FF) or \
           (0x25A0 <= codepoint <= 0x25FF) or \
           (0x2600 <= codepoint <= 0x26FF) or \
           (0x2700 <= codepoint <= 0x27BF) or \
           (0x2B50 <= codepoint <= 0x2B59) or \
           (0x2000 <= codepoint <= 0x206F) or \
           (0x2E00 <= codepoint <= 0x2E7F) or \
           (0x3000 <= codepoint <= 0x303F) or \
           (0x1F300 <= codepoint <= 0x1F5FF):  # Emojis and other pictograms
            return True
        return False
    
    # Remove all characters from the sentence that have been identified as special characters
    cleaned_sentence = ''.join(char for char in sentence if not is_special_character(char))
    return cleaned_sentence


def tokenize_japanese(text):

    tokenizer = MeCab.Tagger()
    tokens = []
    node = tokenizer.parseToNode(text)
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    return tokens

def tokenize_chinese(text):

    tokens = jieba.cut(text)
    return list(tokens)


def tokenize_thai(text):

    tokens = pythainlp.tokenize.word_tokenize(text)
    return tokens

def tokenize_korean(text):

    okt = Okt()
    tokens = okt.morphs(text)
    return tokens


def preprocess(sentence, labels):
    corpus = []
    sentence = sentence.to_list()
    labels = labels.to_list()
    

    chinesecount = 0
    japanesecount = 0
    thaicount = 0
    koreancount = 0
    othercount = 0
    russiancount = 0
    for i in range(len(sentence)):
        processed_sentence =  entferne_sonderzeichen(str(sentence[i].lower())) #entferne_sonderzeichen(sentence[i])
        chinesecount = 0
        japanesecount = 0
        thaicount = 0
        koreancount = 0
        othercount = 0
        russiancount = 0
        #0.2 is the percentage of the sentence that will be randomly selected to determine the language
        randcount = max(int(len(processed_sentence) * 0.8), 1) if len(processed_sentence) > 0 else 0
        
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
            elif 1024 <= ord(processed_sentence[randchar]) <= 1279:
                russiancount += 1
            else:
                othercount += 1
        
        if max(thaicount, japanesecount, chinesecount, koreancount,othercount) == thaicount:
            thai_pattern = r'[^\u0E00-\u0E7F\s,.!?;:-]+'
            processed_sentence = re.sub(thai_pattern, '', processed_sentence)
            tokens = tokenize_thai(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == japanesecount:
            pattern = r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s,.!?;:-]+'
            processed_sentence = re.sub(pattern, '', processed_sentence)
            tokens = tokenize_japanese(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == chinesecount:
            pattern = r'[^\u3400-\u4DBF\u4E00-\u9FFF\u20000-\u2A6DF\s,.!?;:-]+'
            processed_sentence = re.sub(pattern, '', processed_sentence)
            tokens = tokenize_chinese(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,othercount) == koreancount:
            korean_pattern = r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\uD7B0-\uD7FF\s,.!?;:-]+'
            processed_sentence = re.sub(korean_pattern, '', processed_sentence)
            tokens = tokenize_korean(processed_sentence)
        elif max(thaicount, japanesecount, chinesecount, koreancount,russiancount,othercount) == russiancount:
            pattern = r'[^\u0410-\u044F\u0401\u0451\s,.!?;:-]'
            processed_sentence = re.sub(pattern, '', processed_sentence)
            tokens = nltk.word_tokenize(processed_sentence,language='russian')
        
        else:
            #other languages are quiet similar to english so we can use the nltk tokenizer
            
            #example for english stemming
            #stemmer = SnowballStemmer("english")
            #tokens = stem_text(tokens, stemmer)
            
            tokens = nltk.word_tokenize(processed_sentence)
        processed_sentence = ' '.join(tokens)

        corpus.append(processed_sentence)
        
    

    sentence = pd.Series(corpus)
    labels = pd.Series(labels)
    return sentence, labels