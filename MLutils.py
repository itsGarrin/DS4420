from statistics import mode

import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, losses
from keras.models import Model
from scipy.stats import multivariate_normal, binom


class DecisionTree:
    def __init__(self, criterion="entropy", regression=False, max_depth=10, min_instances=1, target_impurity=0.01):
        self.criterion = criterion
        self.regression = regression
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.target_impurity = target_impurity
        self.tree = None

    def fit(self, train):
        self.tree = self.__build_tree(train)

    def predict(self, test):
        return self.__predict(test)

    def __build_tree(self, train, depth=0):
        if self.regression:
            return self.__build_tree_regression(train, depth)
        else:
            return self.__build_tree_classification(train, depth)

    def __build_tree_regression(self, train, depth):
        if depth == self.max_depth:
            return self.__leaf_regression(train)
        elif len(train) <= self.min_instances:
            return self.__leaf_regression(train)
        elif self.__impurity(train) <= self.target_impurity:
            return self.__leaf_regression(train)
        else:
            depth += 1
            best_split = self.__best_split(train)
            left = train[train[best_split[0]] <= best_split[1]]
            right = train[train[best_split[0]] > best_split[1]]
            return [best_split[0], best_split[1], self.__build_tree_regression(left, depth),
                    self.__build_tree_regression(right, depth)]

    def __build_tree_classification(self, train, depth):
        if depth == self.max_depth:
            return self.__leaf_classification(train)
        elif len(train) <= self.min_instances:
            return self.__leaf_classification(train)
        elif self.__impurity(train) <= self.target_impurity:
            return self.__leaf_classification(train)
        else:
            depth += 1
            best_split = self.__best_split(train)
            left = train[train[best_split[0]] <= best_split[1]]
            right = train[train[best_split[0]] > best_split[1]]
            return [best_split[0], best_split[1], self.__build_tree_classification(left, depth),
                    self.__build_tree_classification(right, depth)]

    def __leaf_regression(self, train):
        return train.iloc[:, -1].mean()

    def __leaf_classification(self, train):
        return mode(train.iloc[:, -1])

    def __best_split(self, train):
        best_feature = None
        best_value = None
        best_score = None
        for feature in train.columns[:-1]:
            for value in train[feature].unique():
                score = self.__score(train, feature, value)
                if best_score is None or score < best_score:
                    best_feature = feature
                    best_value = value
                    best_score = score
        return best_feature, best_value

    def __score(self, train, feature, value):
        left = train[train[feature] <= value]
        right = train[train[feature] > value]
        return self.__impurity(left) * len(left) + self.__impurity(right) * len(right)

    def __impurity(self, train):
        if self.criterion == "entropy":
            return self.__entropy(train)
        elif self.criterion == "gini":
            return self.__gini(train)
        elif self.criterion == "mse":
            return self.__mse(train)
        else:
            raise ValueError("Invalid criterion")

    def __entropy(self, train):
        if len(train) == 0:
            return 0
        else:
            p = train.iloc[:, -1].value_counts() / len(train)
            return -sum(p * np.log2(p))

    def __gini(self, train):
        if len(train) == 0:
            return 0
        else:
            p = train.iloc[:, -1].value_counts() / len(train)
            return 1 - sum(p ** 2)

    def __mse(self, train):
        if len(train) == 0:
            return 0
        else:
            return np.var(train.iloc[:, -1])

    def __predict(self, test):
        return test.apply(self.__predict_row, axis=1)

    def __predict_row(self, row):
        return self.__predict_row_helper(row, self.tree)

    def __predict_row_helper(self, row, tree):
        if isinstance(tree, float) or isinstance(tree, int):
            return tree
        else:
            if row[tree[0]] <= tree[1]:
                return self.__predict_row_helper(row, tree[2])
            else:
                return self.__predict_row_helper(row, tree[3])

    def mse(self, test):
        predictions = self.predict(test)
        labels = test.iloc[:, -1]
        return np.mean((predictions - labels) ** 2)

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            predictions = self.predict(validation_fold)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)


# Implement a linear regression class using exact solution
# w = (X^TX)^-1(X^Ty)
class LinearRegression:
    def __init__(self):
        self.w = None
        return

    def fit(self, train):
        X = train.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        y = train.iloc[:, -1]
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.w)

    def mse(self, test):
        y_pred = self.predict(test)
        y = test.iloc[:, -1]
        return np.mean((y - y_pred) ** 2)

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)

            pred = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
            predictions = pred(self.predict(validation_fold))
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)

    def roc_curve(self, train, num_thresholds=51, k=10):
        fold_size = int(len(train) / k)
        matrices = np.zeros(shape=(num_thresholds, 2, 2), dtype=int)
        i = k - 1

        # Shuffle data
        train = train.sample(frac=1).reset_index(drop=True)
        training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
            drop=True)
        validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

        self.fit(training_fold)

        threshold_matrices = []
        validation_predictions = self.predict(validation_fold)
        thresholds = np.linspace(np.min(validation_predictions), np.max(validation_predictions), num_thresholds)

        for threshold in thresholds:
            pred = np.vectorize(lambda y: 1 if y >= threshold else 0)
            predictions = pred(validation_predictions)
            labels = validation_fold[validation_fold.columns[-1]]

            threshold_matrices.append(confusion_matrix(labels, predictions))

        # sum matrices along axis
        matrices = np.add(matrices, threshold_matrices)

        return generate_roc_curve(matrices)


# Build a ridge linear regression class
class RidgeLinearRegression:
    def __init__(self, alpha):
        self.alpha = alpha
        self.weights = None

    def fit(self, train):
        X = train.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        y = train.iloc[:, -1]
        self.weights = np.linalg.inv(X.T @ X + self.alpha * np.identity(X.shape[1])) @ X.T @ y

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

    def mse(self, test):
        y_pred = self.predict(test)
        y = test.iloc[:, -1]
        return np.mean((y_pred - y) ** 2)

    def cross_validate(self, train, k=10):
        fold_size = int(len(train) / k)
        accuracies = []
        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            pred = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
            predictions = pred(self.predict(validation_fold))
            labels = validation_fold[validation_fold.columns[-1]]

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)
        return np.mean(accuracies)


# Build a linear regression class with gradient descent
class LinearGradientDescent:
    def __init__(self, alpha, learning_rate, iterations):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, train):
        X = train.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        y = train.iloc[:, -1]
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            predictions = np.dot(X, self.weights)
            gradient = (X.T.dot(predictions - y) + self.alpha * self.weights)
            self.weights -= self.learning_rate * gradient

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

    def mse(self, test):
        y_pred = self.predict(test)
        y = test.iloc[:, -1]
        return np.mean((y_pred - y) ** 2)

    def cross_validate(self, train, k=10):
        fold_size = int(len(train) / k)
        accuracies = []
        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            pred = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
            predictions = pred(self.predict(validation_fold))
            labels = validation_fold[validation_fold.columns[-1]]

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)
        return np.mean(accuracies)


# Build a logistic regression class with gradient descent
class LogisticGradientDescent:
    def __init__(self, alpha, learning_rate, iterations):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, train):
        X = train.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        y = train.iloc[:, -1]
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            predictions = _sigmoid(X @ self.weights)
            gradient = (X.T @ (predictions - y) + self.alpha * self.weights)
            self.weights -= self.learning_rate * gradient

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        return _sigmoid(X @ self.weights)

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            pred = np.vectorize(lambda y: 1 if y >= 0.5 else 0)
            predictions = pred(self.predict(validation_fold))
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        out = [np.mean(accuracies)]
        if confusion:
            out.append(matrix)

        return out

    def roc_curve(self, train, num_thresholds=51, k=10):
        fold_size = int(len(train) / k)
        matrices = np.zeros(shape=(num_thresholds, 2, 2), dtype=int)
        thresholds = np.linspace(0, 1, num_thresholds)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)

            threshold_matrices = []
            for threshold in thresholds:
                pred = np.vectorize(lambda y: 1 if y >= threshold else 0)
                predictions = pred(self.predict(validation_fold))
                labels = validation_fold[validation_fold.columns[-1]]

                threshold_matrices.append(confusion_matrix(labels, predictions))

            matrices = np.add(matrices, threshold_matrices)
        return generate_roc_curve(matrices)


# Build a perceptron class with normalized weights and counter for mistakes per iteration
class Perceptron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = None
        self.mistakes = None

    def fit(self, train):
        X = train.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        y = train.iloc[:, -1]
        self.weights = np.random.normal(0, 1, X.shape[1])
        self.mistakes = []

        for i in range(100):
            mistakes = 0
            for j in range(X.shape[0]):
                if X[j] @ self.weights <= 0:
                    self.weights += self.learning_rate * X[j]
                    mistakes += 1
            print("Iteration: " + str(i), ", Mistakes: " + str(mistakes))
            self.mistakes.append(mistakes)
            if mistakes == 0:
                break

        return self.weights

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = np.c_[np.ones(X.shape[0]), X]
        return np.sign(X @ self.weights)


class NeuralNetwork:
    def __init__(self, input=8, hiddenNodes=3, outputNodes=8):
        # Set up Architecture of Neural Network
        self.X = None
        self.y = None

        self.input = input
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # Initial Weights
        self.weights1 = np.random.randn(self.input, self.hiddenNodes)  # (8x3)
        self.weights2 = np.random.randn(self.hiddenNodes, self.outputNodes)  # (3x8)

    def train(self, X, y, epochs=1000, learning_rate=1e-3):
        for _ in range(epochs):
            o = self._feed_forward(X)
            self._backward(X, y, o, learning_rate)

    def predict(self, X):
        out = self._feed_forward(X)
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X))
        print("Hidden Layer: \n" + str(self.weights1))
        print("Output: \n" + str(out))
        return out

    def _feed_forward(self, X):
        """
        Calculate the NN inference using feed forward.
        """
        # Weighted sum of inputs and hidden layer
        self.hidden_sum = np.dot(X, self.weights1)

        # Activations of weighted sum
        self.activated_hidden = _sigmoid(self.hidden_sum)

        # Weighted sum between hidden and output
        self.output_sum = np.dot(self.activated_hidden, self.weights2)

        # Final activation of output
        self.activated_output = _sigmoid(self.output_sum)

        return self.activated_output

    def _backward(self, X, y, o, learning_rate):
        """
        Backward propagate through the network
        """
        # Error in output
        self.o_error = y - o

        # Apply derivative of sigmoid to error
        self.o_delta = self.o_error * _sigmoidPrime(o)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.weights2.T)

        # Apply derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error * _sigmoidPrime(self.activated_hidden)

        # Adjustment to first set of weights (input => hidden) with z2_delta
        self.weights1 += X.T.dot(self.z2_delta) * learning_rate

        # Adjustment to second set of weights (hidden => output) with o_delta
        self.weights2 += self.activated_hidden.T.dot(self.o_delta) * learning_rate


class AutoencoderTF(Model):
    def __init__(self, hidden_dim=3, encoding_dim=8):
        super(AutoencoderTF, self).__init__()
        self.hidden_layer = layers.Dense(hidden_dim)
        self.output_layer = layers.Dense(encoding_dim, activation='sigmoid')

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)

    def train(self, X, y, epochs=10000, learning_rate=1e-3):
        self.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                     loss=losses.MeanSquaredError())
        self.fit(X, y, epochs=epochs, shuffle=True, verbose=0)

    def test(self, y):
        predictions = self.predict(y)
        print("Input Data: " + str(y) + "\n" + "Predicted data based on trained weights: " + str(predictions))
        print("Weights: " + str(self.weights))
        return predictions


# Create a GDA class with covariance matrix and mean
class GaussianDiscriminantAnalysis:
    def __init__(self, epsilon=1e-9):
        self.mean = None
        self.covariance = None
        self.epsilon = epsilon

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        self.mean = X.groupby(y).mean()
        self.covariance = X.groupby(y).cov() + self.epsilon

    def predict(self, test):
        X = test.iloc[:, :-1]
        return self.__predict(X).idxmax(axis=1)

    def __predict(self, X):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return pd.DataFrame(predictions)

    def __predict_row(self, row):
        probabilities = []
        for label in self.mean.index:
            probabilities.append(self.__gaussian(row, self.mean.loc[label], self.covariance.loc[label]))
        return pd.Series(probabilities, index=self.mean.index)

    def __gaussian(self, x, mean, covariance):
        return (1 / np.sqrt(np.linalg.det(covariance))) * np.exp(
            -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean))

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            predictions = self.predict(validation_fold)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)


# Create a Bernoulli Naive Bayes class by thresholding against a scalar
class BernoulliNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.priors = None
        self.threshold = None
        self.probs = {}
        self.epsilon = epsilon

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        self.priors = X.groupby(y).size() / len(train)
        self.threshold = X.mean().mean()

        X = X >= self.threshold

        for label in y.unique():
            samples = X[y == label]
            self.probs[label] = samples.mean() + self.epsilon

    def predict(self, test):
        X = test.iloc[:, :-1]
        return pd.DataFrame(self.__predict(X))

    def __predict(self, X):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return predictions

    def __predict_row(self, row):
        probabilities = []
        for label in self.probs:
            probabilities.append(self.__bernoulli(row, self.threshold, self.probs[label], self.priors[label]))
        return pd.Series(probabilities, index=self.probs.keys())

    def __bernoulli(self, x, threshold, prob, prior):
        return np.sum(np.where(x >= threshold, np.log(prob), np.log(1 - prob))) + np.log(prior)

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            predictions = self.predict(validation_fold).idxmax(axis=1)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)

    def roc_curve(self, train, k=10):
        fold_size = int(len(train) / k)

        i = k - 1

        # Shuffle data
        train = train.sample(frac=1).reset_index(drop=True)
        training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
            drop=True)
        validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

        self.fit(training_fold)

        pred = self.predict(validation_fold)
        pred = pred.iloc[:, 0] - pred.iloc[:, 1]
        thresholds = np.unique(pred)
        thresholds = np.sort(thresholds)[::-1]

        matrices = np.zeros(shape=(len(thresholds), 2, 2), dtype=int)

        threshold_matrices = []
        for threshold in thresholds:
            vector = np.vectorize(lambda y: 1 if y >= threshold else 0)
            predictions = vector(pred)
            labels = validation_fold[validation_fold.columns[-1]]

            threshold_matrices.append(confusion_matrix(labels, predictions))

        matrices = np.add(matrices, threshold_matrices)

        return generate_roc_curve(matrices)


# Create a Gaussian Naive Bayes class with mean and variance
class GaussianNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.priors = None
        self.means = None
        self.variances = None
        self.epsilon = epsilon

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        self.priors = X.groupby(y).size() / len(train)
        self.means = X.groupby(y).mean()
        self.variances = X.groupby(y).var() + self.epsilon

    def predict(self, test):
        X = test.iloc[:, :-1]
        return self.__predict(X)

    def __predict(self, X):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return pd.DataFrame(predictions)

    def __predict_row(self, row):
        probabilities = []
        for label in self.means.index:
            probabilities.append(
                self.__gaussian(row, self.means.loc[label], self.variances.loc[label], self.priors[label]))
        return pd.Series(probabilities, index=self.means.index)

    def __gaussian(self, x, mean, variance, prior):
        return np.log(prior) + np.sum(-0.5 * np.log(2 * np.pi * variance) - 0.5 * ((x - mean) ** 2 / variance))

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            predictions = self.predict(validation_fold).idxmax(axis=1)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)

    def roc_curve(self, train, k=10):
        fold_size = int(len(train) / k)

        i = k - 1

        # Shuffle data
        train = train.sample(frac=1).reset_index(drop=True)
        training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
            drop=True)
        validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

        self.fit(training_fold)

        pred = self.predict(validation_fold)
        pred = pred.iloc[:, 1] - pred.iloc[:, 0]
        thresholds = np.unique(pred)
        thresholds = np.sort(thresholds)[::-1]

        matrices = np.zeros(shape=(len(thresholds), 2, 2), dtype=int)

        threshold_matrices = []
        for threshold in thresholds:
            vector = np.vectorize(lambda y: 1 if y >= threshold else 0)
            predictions = vector(pred)
            labels = validation_fold[validation_fold.columns[-1]]

            threshold_matrices.append(confusion_matrix(labels, predictions))

        matrices = np.add(matrices, threshold_matrices)

        return generate_roc_curve(matrices)


class GaussianEM:
    def __init__(self, dim, k, epochs=500, epsilon=1e-9):
        self.means = [np.random.normal(size=dim) for _ in range(k)]
        self.covariances = [np.eye(dim) for _ in range(k)]
        self.weights = np.ones(k) / k
        self.dim = dim
        self.k = k
        self.epsilon = epsilon
        self.epochs = epochs
        self.memberships = None

    def fit(self, X):
        for _ in range(self.epochs):
            self.__e_step(X)
            self.__m_step(X)

        return self.means, self.covariances

    def __e_step(self, X):
        self.memberships = np.zeros((X.shape[0], self.weights.shape[0]))
        for i in range(self.weights.shape[0]):
            self.memberships[:, i] = self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covariances[i])

        self.memberships = self.memberships / self.memberships.sum(axis=1, keepdims=True)

    def __m_step(self, X):
        self.weights = self.memberships.sum(axis=0) / X.shape[0]

        for i in range(self.weights.shape[0]):
            self.means[i] = np.sum(self.memberships[:, i, None] * X, axis=0) / self.memberships[:, i].sum()

            diff = X - self.means[i]
            self.covariances[i] = np.dot(self.memberships[:, i] * diff.T, diff) / self.memberships[:,
                                                                                  i].sum() + self.epsilon


# create a EM class to predict coin flips
class BinomialEM:
    def __init__(self, k, epochs=500, epsilon=1e-9):
        self.probs = np.random.random((k, 1))
        self.weights = np.ones(k) / k
        self.k = k
        self.epsilon = epsilon
        self.epochs = epochs
        self.memberships = None

    def fit(self, X):
        for _ in range(self.epochs):
            self.__e_step(X)
            self.__m_step(X)

        return self.weights, self.probs

    def __e_step(self, X):
        self.memberships = np.zeros((X.shape[0], self.weights.shape[0]))
        for i in range(self.weights.shape[0]):
            self.memberships[:, i] = self.weights[i] * np.prod(binom.pmf(X, 1, self.probs[i]), axis=1)

        self.memberships = self.memberships / self.memberships.sum(axis=1, keepdims=True)

    def __m_step(self, X):
        self.weights = self.memberships.sum(axis=0) / X.shape[0]

        for i in range(self.weights.shape[0]):
            self.probs[i] = np.sum(self.memberships[:, i][:, None] * X, axis=0).sum() / (
                    self.memberships[:, i].sum() * X.shape[1])


# Create an AdaBoost class
class AdaBoost:
    def __init__(self, T=10):
        self.T = T
        self.alphas = None
        self.models = None

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        n = len(X)
        w = np.ones(n) / n
        self.models = []
        self.alphas = []

        for _ in range(self.T):
            model = DecisionTree(max_depth=1)
            model.fit(train, sample_weight=w)
            predictions = model.predict(X)
            error = w[(predictions != y)].sum()
            alpha = 0.5 * np.log((1 - error) / error)
            w = w * np.exp(-alpha * y * predictions)
            w = w / w.sum()
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, test):
        X = test.iloc[:, :-1]
        predictions = np.zeros(len(X))

        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X)

        predictions = np.sign(predictions)
        return predictions

    def cross_validate(self, train, k=10, confusion=False):
        fold_size = int(len(train) / k)
        accuracies = []
        matrix = np.zeros(shape=(2, 2), dtype=int)

        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.fit(training_fold)
            predictions = self.predict(validation_fold)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)


def normalize(data):
    avg = data.mean()
    stdev = data.std()
    return (data - avg) / stdev, avg, stdev


def plot_roc_curve(fpr, tpr):
    """ Plot a ROC curve.
    fpr = false positive rate
    tpr = true positive rate
    """

    # Plot ROC curve
    plt.plot(fpr, tpr)

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Show plot
    plt.show()


def calculate_auc(fpr, tpr):
    """ Calculate area under the curve.
    fpr = false positive rate
    tpr = true positive rate
    return area under the curve
    """

    # Initialize area under the curve
    auc = 0

    # Iterate over each point on the curve
    for i in range(len(fpr) - 1):
        # Calculate the area of the trapezoid
        auc += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2

    return auc


def confusion_matrix(y_true, y_pred):
    """ Generate a confusion matrix.
    y = actual outcomes (0, 1, 2, ...)
    y_pred = predicted outcomes (0, 1, 2, ...)
    return confusion matrix as a numpy array
    """

    # Find unique identifiers
    unique_classes = set(y_true) | set(y_pred)
    n_classes = len(unique_classes)

    # Create matrix (all zeros)
    matrix = np.zeros(shape=(n_classes, n_classes), dtype=int)

    # Pair up each actual outcome with the corresponding prediction
    actual_prediction = list(zip(y_true, y_pred))

    # For each pair, increment the correct position in the matrix
    for i, j in actual_prediction:
        matrix[i, j] += 1

    return matrix


def generate_roc_curve(matrices):
    """ Generate a ROC curve.
    y = actual outcomes (0, 1, 2, ...)
    y_pred = predicted outcomes (0, 1, 2, ...)
    return false positive rates and true positive rates
    """

    # Initialize false positive rates, true positive rates, and thresholds
    fpr = []
    tpr = []

    # Iterate over each threshold
    for matrix in matrices:
        # Initialize true positives, false positives, true negatives, and false negatives
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        # Iterate over each confusion matrix
        # Calculate true positives, false positives, true negatives, and false negatives
        tp += matrix[1, 1]
        fp += matrix[0, 1]
        tn += matrix[0, 0]
        fn += matrix[1, 0]

        # Calculate false positive rate and true positive rate
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr


def _sigmoid(s):
    return 1 / (1 + np.exp(-s))


def _sigmoidPrime(s):
    return s * (1 - s)
