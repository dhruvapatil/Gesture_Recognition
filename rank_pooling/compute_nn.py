import numpy as np
import os
from os.path import join
from sklearn.svm import SVC
from model.NeuralNetwork import NeuralNets, NeuralNet2
from dependencies.Graphs import plot_confusion_matrix
from classifier.ModelFunctions import (classify, calculate_confusion_matrix)

main_folder = 'index_log_files'
features_train = np.load(join(main_folder, 'features_train.npy'))
labels_train = map(int, np.load(join(main_folder, 'labels_train.npy')))

features_val = np.load(join(main_folder, 'features_val.npy'))
labels_val = map(int, np.load(join(main_folder, 'labels_val.npy')))

features_valid = np.load(join(main_folder, 'features_valid.npy'))

def generate_confusion_matrix(clf, title):
    train_cm, test_cm = classify(features_train, labels_train, features_val, labels_val, clf)
    #train_cm_right, test_cm_right = classify(X_train_right, y_train_right, X_val_right, y_val_right, clf)

    np.save(join(main_folder, title + '_CM_train_inter.npy'), train_cm)
    np.save(join(main_folder, title + '_CM_test_inter.npy'), test_cm)
    '''
    np.save(join(main_folder, title+'_CM_train_left.npy'), train_cm_left)
    np.save(join(main_folder, title+'_CM_test_left.npy'), test_cm_left)
    np.save(join(main_folder, title+'_CM_train_right.npy'), train_cm_right)
    np.save(join(main_folder, title + '_CM_test_right.npy'), test_cm_right)
    '''

    train_acc, test_acc = np.mean(train_cm.diagonal()), np.mean(test_cm.diagonal())
    #train_acc_right, test_acc_right = np.mean(train_cm_right.diagonal()), np.mean(test_cm_right.diagonal())

    print title, train_acc, test_acc#, train_acc_right, test_acc_right


print features_train.shape, len(labels_train)
print features_val.shape, len(labels_val)
print features_valid.shape
'''
left_train_samples = int(0.9*features_left.shape[0])
right_train_samples = int(0.9*feature_right.shape[0])

X_train_left = features_left[:left_train_samples, :]
y_train_left = labels_left[:left_train_samples]

X_val_left = features_left[left_train_samples:, :]
y_val_left = labels_left[left_train_samples:]

X_train_right = feature_right[:right_train_samples, :]
y_train_right = labels_right[:right_train_samples]

X_val_right = feature_right[right_train_samples:, :]
y_val_right = labels_right[right_train_samples:]
'''

nn = NeuralNets(features_train, features_val, labels_train, labels_val, epochs=3000, n_neurons=100, alpha=0.001, batch_size=2490)
#nn = NeuralNet2(features_train, features_val, labels_train, labels_val, epochs=3000, n_neurons_1=200, n_neurons_2=250, alpha=0.001, batch_size=12450)

#from sklearn.neural_network import MLPClassifier
#clf = MLPClassifier(hidden_layer_sizes=(200,200)) #taking all default values the package provides
#generate_confusion_matrix(clf, title='MLPerceptron')

'''
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_val)
y_actual = labels_val

print 'Test accuracy is: ', sum(y_pred==y_actual)/len(labels_val)

proba_matrix=  clf.predict_proba(features_valid)
print 'shape of proba matrix: ', proba_matrix.shape

np.save('files/probability_values_both_arms.npy', proba_matrix)

#from sklearn.linear_model import Perceptron
#clf = Perceptron()
#generate_confusion_matrix(clf, title='LinearPerceptron')
'''


'''
classes_left = list(set(labels_left))
classes_right = list(set(labels_right))
plot_confusion_matrix(train_cm_left, classes_left, title='training_CM_Left')
plot_confusion_matrix(test_cm_left, classes_left, title='testing_CM_Left')
plot_confusion_matrix(train_cm_right, classes_right, title='training_CM_Right')
plot_confusion_matrix(test_cm_right, classes_right, title='testing_CM_Right')
from collections import Counter

print Counter(labels_left)
print Counter(labels_right)
'''