import pandas as pd

def preprocess(sentence, labels):
    '''
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting, 
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    '''
    
    
    
    if isinstance(sentence, pd.Series):
        sentence = sentence.iloc[0]  # Annahme, dass die Serie nur einen Satz enthält
    
    # Initialisierung einer leeren String-Variable für die Sammlung der Ausgabe
    output = ''
    
    # Iteration über jeden Charakter im Satz
    
    for i in range(0,5):
        chara = sentence[i]
        if ord(chara) > 1000:
            print(f"Character {chara} is  an asian character")


    # Rückgabe des ursprünglichen Satzes und Labels ohne Änderung
    return sentence, labels
