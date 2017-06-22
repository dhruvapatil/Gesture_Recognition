import numpy as np
import tensorflow as tf
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from classifier.ModelFunctions import calculate_confusion_matrix

class NeuralNets:
    def __init__(self, train_X, test_X, train_y, test_y, epochs, n_neurons, alpha, batch_size):

        index_list = self.get_label_index_distribution(train_y)
        classes = set(train_y)
        train_X, test_X, train_y, test_y = self.processed_data(train_X, test_X, train_y, test_y)

        # Layer's sizes
        x_size = train_X.shape[1]
        h_size = n_neurons  # Number of hidden nodes
        y_size = train_y.shape[1]
        alpha = alpha

        print train_X.shape, test_X.shape, train_y.shape, test_y.shape

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        w_1 = self.init_weights((x_size, h_size))
        w_2 = self.init_weights((h_size, y_size))

        # Forward propagation
        yhat = self.forwardprop(X, w_1, w_2)
        predict = tf.argmax(yhat, dimension=1)

        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))
        updates = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

        # Run SGD
        sess = tf.Session()
        initial = tf.global_variables_initializer()  # global_variables_initializer()
        sess.run(initial)

        accuracy_train = []
        accuracy_test = []
        cost_list= []


        for epoch in range(epochs):
            #Train for a batch of 36 with 6 random entries from each class
            X_train, y_train = self.get_batch(train_X, train_y, index_list, batch_size=batch_size)

            sess.run(updates, feed_dict={X: X_train, y: y_train})
            print X_train.shape, y_train.shape

            train_accuracy = np.mean(np.argmax(y_train, axis=1) ==
                                     sess.run(predict, feed_dict={X: X_train, y: y_train}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))
            #print "Testing results: ", sess.run(predict, feed_dict={X: test_X, y: test_y})

            cost_list.append(sess.run(cost, feed_dict={X:X_train, y:y_train}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            accuracy_train.append((100. * train_accuracy))
            accuracy_test.append((100. * test_accuracy))
            if(epoch%10==0):

                #cm = confusion_matrix(np.argmax(y_train, axis=1), sess.run(predict, feed_dict={X: X_train, y: y_train}))
                #plot_confusion_matrix(cm, classes, epoch, flag=0)
                cm = calculate_confusion_matrix(np.argmax(test_y, axis=1), sess.run(predict, feed_dict={X: test_X, y: test_y}))
                print 'Test Accuracy (to confirm) :', np.mean(cm.diagonal())

        sess.close()


    def get_label_index_distribution(self, y_train):
        classes = len(set(y_train))
        index_list = []
        for k in range(1,(classes+1)):
            data_filter = [i for (i, j) in enumerate(y_train) if j == k]  # and train_data[i].shape[0] >= (window_size)]
            index_list.append(data_filter)
        return index_list


    def processed_data(self, train_X, test_X, train_y, test_y):
        all_X_train = self.add_bias(train_X)
        all_X_test = self.add_bias(test_X)

        print all_X_train.shape, all_X_test.shape
        lb = LabelBinarizer()
        lb.fit(train_y)

        y_train = lb.transform(train_y)
        y_test = lb.transform(test_y)
        print y_train.shape, y_test.shape
        return all_X_train, all_X_test, y_train, y_test


    def add_bias(self, data):
        N, M = data.shape
        all_data = np.ones((N, M + 1))
        all_data[:, 1:] = data
        return all_data


    def init_weights(self, shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        return tf.Variable(weights)


    def forwardprop(self, X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat


    def get_batch(self, train_X, train_y, index_list, batch_size):
        from random import sample
        classes = train_y.shape[1]
        unit = batch_size / classes

        index = []
        for i in index_list:
            index.append(sample(i, unit))
        index = list(chain(*index))

        X_train = train_X[index, :]
        y_train = train_y[index,:]
        return X_train, y_train


class NeuralNet2:
    def __init__(self, train_X, test_X, train_y, test_y, epochs, n_neurons_1, n_neurons_2, alpha, batch_size):

        index_list = self.get_label_index_distribution(train_y)
        print 'index list is: ', index_list
        train_X, test_X, train_y, test_y = self.processed_data(train_X, test_X, train_y, test_y)

        # Layer's sizes
        x_size = train_X.shape[1]
        h1_size = n_neurons_1  # Number of hidden nodes
        h2_size = n_neurons_2
        y_size = train_y.shape[1]
        alpha = alpha

        print train_X.shape, test_X.shape, train_y.shape, test_y.shape

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size])
        y = tf.placeholder("float", shape=[None, y_size])

        # Weight initializations
        weights = {
            'h1': tf.Variable(tf.random_normal((x_size, h1_size))),
            'h2': tf.Variable(tf.random_normal((h1_size, h2_size))),
            'out': tf.Variable(tf.random_normal((h2_size, y_size)))
        }

        # Bias initializations
        bias = {
            'b1': tf.Variable(tf.random_normal([h1_size])),
            'b2': tf.Variable(tf.random_normal([h2_size])),
            'out': tf.Variable(tf.random_normal([y_size]))
        }

        yhat = self.multilayer_perceptron(X, weights, bias)
        predict = tf.argmax(yhat, dimension=1)

        # Backward propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))
        updates = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

        # Run SGD
        sess = tf.Session()
        initial = tf.initialize_all_variables()#global_variables_initializer()  # global_variables_initializer()
        sess.run(initial)

        accuracy_train = []
        accuracy_test = []
        cost_list = []

        for epoch in range(epochs):
            # Train for a batch of 36 with 6 random entries from each class
            X_train, y_train = self.get_batch(train_X, train_y, index_list, batch_size=batch_size)

            sess.run(updates, feed_dict={X: X_train, y: y_train})


            #print 'shapes are: ', X_train.shape, y_train.shape


            train_accuracy = np.mean(np.argmax(y_train, axis=1) ==
                                     sess.run(predict, feed_dict={X: X_train, y: y_train}))
            test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                    sess.run(predict, feed_dict={X: test_X, y: test_y}))
            # print "Testing results: ", sess.run(predict, feed_dict={X: test_X, y: test_y})

            cost_list.append(sess.run(cost, feed_dict={X: X_train, y: y_train}))

            if ((epoch+1) % 100 == 0):
                print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                      % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
            accuracy_train.append((100. * train_accuracy))
            accuracy_test.append((100. * test_accuracy))
            #if (epoch % 1000 == 0):
            #    plot_cost_function(cost_list, (epoch + 1))
            #    plot_accuracy(accuracy_train, accuracy_test, (epoch + 1))

            #if (epoch%10000 == 0):
            #    alpha = alpha/10

        print train_accuracy
        print test_accuracy
        sess.close()

    def get_label_index_distribution(self, y_train):
        classes = len(set(y_train))
        print 'classes is: ', classes
        index_list = []
        for k in range(1, (classes+1)):
            data_filter = [i for (i, j) in enumerate(y_train) if j == k]  # and train_data[i].shape[0] >= (window_size)]
            index_list.append(data_filter)
        return index_list

    def processed_data(self, train_X, test_X, train_y, test_y):
        all_X_train = self.add_bias(train_X)
        all_X_test = self.add_bias(test_X)

        print all_X_train.shape, all_X_test.shape
        lb = LabelBinarizer()
        lb.fit(train_y)

        y_train = lb.transform(train_y)
        y_test = lb.transform(test_y)
        print y_train.shape, y_test.shape
        return all_X_train, all_X_test, y_train, y_test

    def add_bias(self, data):
        N, M = data.shape
        all_data = np.ones((N, M + 1))
        all_data[:, 1:] = data
        return all_data


    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    def get_batch(self, train_X, train_y, index_list, batch_size):
        from random import sample
        classes = train_y.shape[1]
        unit = batch_size / classes

        index = []
        for i in index_list:
            index.append(sample(i, unit))
        index = list(chain(*index))

        X_train = train_X[index, :]
        y_train = train_y[index, :]
        return X_train, y_train