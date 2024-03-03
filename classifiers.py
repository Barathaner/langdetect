#classifiers.py
from sklearn.naive_bayes import MultinomialNB
from utils import toNumpyArray

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from hmmlearn import hmm
from sklearn.ensemble import RandomForestClassifier

def applyKNN(X_train, y_train, X_test):
    '''
    Task: Given some features train a KNN classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applySVM(X_train, y_train, X_test):
    '''
    Task: Given some features train a SVM classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = SVC()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applyRandomForest(X_train, y_train, X_test):
    '''
    Task: Given some features train a Random Forest classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = RandomForestClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


'''
from collections import Counter
def voting(NB, KNN, SVM, RF):
    """
    Perform voting among the predictions of four classifiers and return the most common prediction.
    In the event of a tie, default to the prediction of the Random Forest classifier.

    Parameters:
    - NB (array-like): Predictions from Naive Bayes classifier.
    - KNN (array-like): Predictions from KNN classifier.
    - SVM (array-like): Predictions from SVM classifier.
    - RF (array-like): Predictions from Random Forest classifier.

    Returns:
    - array: The final ensemble predictions based on voting.
    """

    # Combine predictions from all classifiers
    all_predictions = [NB, KNN, SVM, RF]

    # Transpose the list to have predictions for each sample in one place
    all_predictions = zip(*all_predictions)

    final_predictions = []

    # Iterate through predictions for each sample
    for predictions in all_predictions:
        # Count occurrences of each prediction
        vote_counts = Counter(predictions)

        # Sort the predictions by count (descending order)
        sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)

        # Check if there's a tie for the most common prediction
        if len(sorted_votes) > 1 and sorted_votes[0][1] == sorted_votes[1][1]:
            # If there's a tie, default to the prediction of the Random Forest classifier
            for i, prediction in enumerate(predictions):
                if prediction == RF[i]:  # Assuming RF is the prediction from the Random Forest classifier
                    final_prediction = prediction
                    break
        else:
            # Otherwise, select the most common prediction
            final_prediction = sorted_votes[0][0]

        # Append the final prediction to the list
        final_predictions.append(final_prediction)

    return final_predictions

'''

def voting(NB, SVM, RF):
    """
    Perform predictions based on specified rules:
    1. Default to Random Forest prediction.
    2. Predict Chinese only if Naive Bayes predicts Chinese.
    3. Predict Latin only if SVM predicts Latin.

    Parameters:
    - NB (array-like): Predictions from Naive Bayes classifier.
    - SVM (array-like): Predictions from SVM classifier.
    - RF (array-like): Predictions from Random Forest classifier.

    Returns:
    - array: The final predictions based on the specified rules.
    """

    final_predictions = []

    for nb_pred, svm_pred, rf_pred in zip(NB, SVM, RF):
        if nb_pred == 'Chinese': #Because NB does a better job at predicting Chinese
            final_predictions.append('Chinese')
        elif svm_pred == 'Latin':
            final_predictions.append('Latin') #SVM does better at latin
        else:
            final_predictions.append(rf_pred)

    return final_predictions 
