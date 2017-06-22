import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_accuracy(act, pred):
    correct = np.sum(act == pred)
    total = pred.shape[0]
    accuracy = float(correct) / total
    return accuracy



def calculate_confusion_matrix(act, pred, normalize=True):
    def normalize_cm(cm):
        cm = np.array(cm, dtype=float)
        total = [float(sum(row)) for row in cm]
        rows, cols = cm.shape[0], cm.shape[1]
        for i in range(rows):
            for j in range(cols):
                cm[i][j] = cm[i][j]/total[i]
        return cm

    cm = confusion_matrix(act, pred)
    #print 'before normalization'
    #print cm
    if normalize:
        cm = normalize_cm(cm)
        #print 'after normalization: '
        #print cm

    #accuracy = np.mean(cm.diagonal())
    return cm#accuracy



def classify(X_train, y_train, X_test, y_test, classifier):
    # print 'In the classify module: ', X_train.shape, X_test.shape, len(y_train), len(y_test)
    clf = classifier
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = calculate_confusion_matrix(y_train, y_train_pred)#calculate_accuracy(y_train, y_train_pred)
    test_accuracy = calculate_confusion_matrix(y_test, y_test_pred)#calculate_accuracy(y_test, y_test_pred)
    return train_accuracy, test_accuracy