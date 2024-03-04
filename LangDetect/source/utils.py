from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.decomposition import PCA
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Für 3D-Plots
from collections import Counter

def compute_features(X_train, 
                     X_test, 
                     analyzer='char', 
                     max_features=None):
  '''
  Task: Compute a matrix of token counts given a corpus. 
        This matrix represents the frecuency any pair of tokens appears
        together in a sentence.
 
  Input: X_train -> Train sentences
         X_test -> Test sentences
         analyzer -> Granularity used to process the sentence 
                    Values: {word, char}
         tokenizer -> Callable function to apply to the sentences before compute.
  
  Output: unigramFeatures: Cout matrix
          X_unigram_train_raw: Features computed for the Train sentences
          X_unigram_test_raw: Features computed for the Test sentences 
  '''
  
  unigramVectorizer = CountVectorizer(analyzer=analyzer,
                                      max_features=max_features,
                                      ngram_range=(1,1))
  
  X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
  X_unigram_test_raw = unigramVectorizer.transform(X_test)
  unigramFeatures = unigramVectorizer.get_feature_names_out()
  return unigramFeatures, X_unigram_train_raw, X_unigram_test_raw
    


def compute_coverage(features, split, analyzer='char'):
  '''
  Task: Compute the proportion of a corpus that is represented by
        the vocabulary. All non covered tokens will be considered as unknown
        by the classifier.
  
  Input: features -> Count matrix
         split -> Set of sentence 
         analyzer -> Granularity level {'word', 'char'}
  
  Output: proportion of covered tokens
  '''
  total = 0.0
  found = 0.0
  for sent in split:
    #The following may be affected by your preprocess function. Modify accordingly
    sent = sent.split(' ') if analyzer == 'word' else list(sent)
    total += len(sent)
    for token in sent:
      if token in features:
        found += 1.0
  return found / total

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    '''
    Task: Cast different types into numpy.ndarray
    Input: data ->  ArrayLike object
    Output: numpy.ndarray object
    '''
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    print(data_type)
    return None    
  
def normalizeData(train, test):
    '''
    Task: Normalize data to train classifiers. This process prevents errors
          due to features with different scale
    
    Input: train -> Train features
           test -> Test features

    Output: train_result -> Normalized train features
            test_result -> Normalized test features
    '''
    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
    return train_result, test_result


def plot_F_Scores(y_test, y_predict):
    '''
    Task: Compute the F1 score of a set of predictions given
          its reference

    Input: y_test: Reference labels 
           y_predict: Predicted labels

    Output: Print F1 score
    '''
    f1_micro = f1_score(y_test, y_predict, average='micro')
    f1_macro = f1_score(y_test, y_predict, average='macro')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))

def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    '''
    Task: Given a set of reference and predicted labels plot its confussion matrix
    
    Input: y_test ->  Reference labels
           y_predict -> Predicted labels
           color -> [Optional] Color used for the plot
    
    Ouput: Confussion Matrix plot
    '''
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sn.set(font_scale=0.8) # for label size
    sn.set(rc={'figure.figsize':(15, 15)})
    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size
    plt.show()


def output_incorrect_predictions(X_data, y_true, y_pred):
    """
    Outputs the sentences and their actual and predicted languages for those instances where the prediction was wrong.
    
    Parameters:
    - X_data: The original sentences (pandas Series or similar iterable).
    - y_true: The actual labels (pandas Series or similar iterable).
    - y_pred: The predicted labels (list or similar iterable).
    """
    print('Incorrect Predictions:')
    print('======================')
    for sentence, actual, predicted in zip(X_data, y_true, y_pred):
        if actual != predicted:
            print(f"Sentence: {sentence}")
            print(f"Actual Language: {actual}, Predicted Language: {predicted}")
            print('----------------------')

# Place this function call after you have predictions from your classifier
# Make sure to pass the original test set sentences (X_test before any transformations if preprocess changes them), actual labels (y_test), and predicted labels (y_predict)




def plotPCA3D(x_train, x_test, y_test, langs):
    '''
    Task: Given train features train a PCA dimensionality reduction
          (3 dimensions) and plot the test set according to its labels.
    
    Input: x_train -> Train features
           x_test -> Test features
           y_test -> Test labels
           langs -> Set of language labels

    Output: Print the amount of variance explained by the 3 first principal components.
            Plot PCA results by language in 3D
    '''
    pca = PCA(n_components=3)
    pca.fit(toNumpyArray(x_train))
    pca_test = pca.transform(toNumpyArray(x_test))
    print('Variance explained by PCA:', pca.explained_variance_ratio_)
    y_test_list = np.asarray(y_test.tolist())

    # Erzeuge eine 3D-Figur
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    base_cmap_b = plt.cm.get_cmap('tab20b', 20)
    base_cmap_c = plt.cm.get_cmap('tab20c', 20)
    colors_b = [base_cmap_b(i) for i in range(20)]
    colors_c = [base_cmap_c(i) for i in range(20)]
    colors = colors_b + colors_c  # Kombiniere beide Paletten

    for idx, lang in enumerate(langs):
        pca_x = pca_test[y_test_list == lang, 0]
        pca_y = pca_test[y_test_list == lang, 1]
        pca_z = pca_test[y_test_list == lang, 2]  # Dritte Dimension
        ax.scatter(pca_x, pca_y, pca_z, color=colors[idx % len(colors)], label=lang)

    ax.legend(loc="best", bbox_to_anchor=(1.05, 1))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3D PCA of Test Set by Language')
    plt.tight_layout()
    plt.show()



def plotPCA(x_train, x_test,y_test, langs):
    '''
    Task: Given train features train a PCA dimensionality reduction
          (2 dimensions) and plot the test set according to its labels.
    
    Input: x_train -> Train features
           x_test -> Test features
           y_test -> Test labels
           langs -> Set of language labels

    Output: Print the amount of variance explained by the 2 first principal components.
            Plot PCA results by language
            
    '''
    pca = PCA(n_components=2)
    pca.fit(toNumpyArray(x_train))
    pca_test = pca.transform(toNumpyArray(x_test))
    print('Variance explained by PCA:', pca.explained_variance_ratio_)
    y_test_list = np.asarray(y_test.tolist())

    base_cmap_b = plt.cm.get_cmap('tab20b', 20)
    base_cmap_c = plt.cm.get_cmap('tab20c', 20)
    colors_b = [base_cmap_b(i) for i in range(20)]
    colors_c = [base_cmap_c(i) for i in range(20)]
    colors = colors_b + colors_c  # Kombiniere beide Paletten

    for idx, lang in enumerate(langs):
        pca_x = pca_test[y_test_list == lang, 0]
        pca_y = pca_test[y_test_list == lang, 1]
        # Verwende den Modulo-Operator, um sicherzustellen, dass der Index innerhalb der Länge der Farbliste bleibt
        plt.scatter(pca_x, pca_y, color=colors[idx % len(colors)], label=lang)
    
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Test Set by Language')
    plt.tight_layout()
    plt.show()
