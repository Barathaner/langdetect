from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from utils import toNumpyArray

def applyNaiveBayes(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applySVM(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    # Hier können Sie den kernel-Parameter an Ihre Bedürfnisse anpassen
    clf = SVC(kernel='linear')
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

def applyKNN(X_train, y_train, X_test, n_neighbors=3):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict

# Beispiel für die Verwendung:
# Angenommen, X_train, y_train, X_test sind bereits definiert
# y_pred_nb = applyNaiveBayes(X_train, y_train, X_test)
# y_pred_svm = applySVM(X_train, y_train, X_test)
# y_pred_knn = applyKNN(X_train, y_train, X_test, n_neighbors=5)
