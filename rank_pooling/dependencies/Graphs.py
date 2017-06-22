import numpy as np
from os.path import join
from os.path import exists
from os import mkdir
from itertools import product
import matplotlib.pyplot as plt

#des_directory = '/s/chopin/k/grad/dkpatil/results/move_vs_still/classifiers/'
cm_des_dir = 'C:/Users/patil/PycharmProjects/ChaLearn/files'

#des_directory = '/home/dhruva/neural_net_studies/results/17_joint_2_10000/'
#if not exists(join(des_directory, 'cost')): mkdir(join(des_directory, 'cost'))
#if not exists(join(des_directory, 'epoch')): mkdir(join(des_directory, 'epoch'))

def plot_cost_function(cost_list, epoch):
    x = np.arange(epoch)
    plt.xlabel('epochs')
    plt.ylabel('train_cost')
    plt.plot(x, cost_list)
    plt.savefig((join(des_directory, 'cost', (str(epoch) + '.png'))))
    plt.close()


def plot_accuracy(train_accuracy, test_accuracy, epoch):
    x = np.arange(epoch)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(x, train_accuracy)
    plt.plot(x, test_accuracy)
    plt.legend(['training accuracy', 'testing accuracy'], loc=4)
    plt.savefig((join(des_directory, 'epoch', (str(epoch) + '.png'))))
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(join(cm_des_dir, (title+'.png')))
    plt.close()
    #plt.show()


def plot_subplots(array, colors):
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
    ax1.plot(array[0], 'o', color = colors[0])
    ax1.set_title('RA move up')
    ax2.plot(array[1], 'o', color=colors[1])
    ax2.set_title('RA move down')
    ax3.plot(array[2], 'o', color=colors[2])
    ax3.set_title('RA move right')
    ax4.plot(array[3], 'o', color=colors[3])
    ax4.set_title('RA move left')
    ax5.plot(array[4], 'o', color=colors[4])
    ax5.set_title('RA move front')
    ax6.plot(array[5], 'o', color=colors[5])
    ax6.set_title('RA move back')
    plt.show()
