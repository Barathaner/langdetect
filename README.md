We are the coolest


venv) PS C:\Users\User\Desktop\langdetect\LangDetect\source> python langdetect.py -i ../data/dataset.csv -a word -v 8000
========
Languages {'French', 'Dutch', 'Hindi', 'Urdu', 'Arabic', 'Latin', 'Japanese', 'Swedish', 'Persian', 'Indonesian', 'Tamil', 'Russian', 'English', 'Chinese', 'Spanish', 'Pushto', 'Estonian', 'Korean', 'Turkish', 'Thai', 'Portugese', 'Romanian'}
========
language
Estonian      1000
Swedish       1000
English       1000
Russian       1000
Romanian      1000
Persian       1000
Pushto        1000
Spanish       1000
Hindi         1000
Korean        1000
Chinese       1000
French        1000
Portugese     1000
Indonesian    1000
Urdu          1000
Latin         1000
Turkish       1000
Japanese      1000
Dutch         1000
Tamil         1000
Thai          1000
Arabic        1000
Name: count, dtype: int64
========
Split sizes:
Train: 17600
Test: 4400
========
train data
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\User\AppData\Local\Temp\jieba.cache
Loading model cost 0.583 seconds.
Prefix dict has been built successfully.
test data
========
Number of tokens in the vocabulary: 8000
Coverage:  0.36045203220486954
========
========
Prediction Results:
F1: 0.9779545454545454 (micro), 0.9785874252067778 (macro), 0.9787911701959092 (weighted)
========
========
PCA and Explained Variance:
Variance explained by PCA: [0.05839863 0.02498133]
========

