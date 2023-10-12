from collections import Counter
from math import log2

import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class DecisionTree:
    def __init__(self, criterion, regression, max_depth=None, min_instances=2, target_impurity=0.0):
        self.criterion = criterion
        self.regression = regression
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.target_impurity = target_impurity

    # Function of dtree taking following parameters
    def fit(self, train, depth=0):
        num_instances = len(train)
        num_classes = len(train[train.columns[-1]].unique())
        class_counts = train[train.columns[-1]]

        # Checking max_depth and min_depth
        if num_classes == 1 or (
                self.max_depth is not None and depth == self.max_depth) or num_instances < self.min_instances:
            return None, None, num_instances, self.regression(class_counts), 0, depth, None, None

        best_col, best_v, best_mae = self.__best_split(train, train.columns[-1], self.criterion)

        left_split = train[train[best_col] <= best_v]
        right_split = train[train[best_col] > best_v]

        if left_split.empty or right_split.empty:
            left_split = train[train[best_col] == best_v]
            right_split = train[train[best_col] != best_v]

        # Checking for target impurity
        if best_mae <= self.target_impurity:
            left = None, None, len(left_split), self.regression(left_split[train.columns[-1]]), 0, depth + 1, None, None
            right = None, None, len(right_split), self.regression(
                right_split[train.columns[-1]]), 0, depth + 1, None, None
            return best_col, best_v, num_instances, self.regression(class_counts), best_mae, depth, left, right

        # Recursing
        left = self.fit(left_split, depth + 1)
        right = self.fit(right_split, depth + 1)

        return best_col, best_v, num_instances, self.regression(class_counts), best_mae, depth, left, right

    def predict(self, model, data):
        predictions = []
        for _, row in data.iterrows():
            predictions.append(self.__predict_row(model, row))
        return predictions

    # cross validation to determine overall validation error
    # Returning avg. accuracies
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

            model = self.fit(training_fold)
            predictions = self.predict(model, validation_fold)
            labels = validation_fold[validation_fold.columns[-1]]

            matrix += confusion_matrix(labels, predictions)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)

        if confusion:
            return np.mean(accuracies), matrix
        else:
            return np.mean(accuracies)

    def mse(self, model, test):
        y_pred = self.predict(model, test)
        y = test.iloc[:, -1]
        return np.mean((y - y_pred) ** 2)

    def __wavg(self, cnt1, cnt2, measure):
        tot1 = len(cnt1)
        tot2 = len(cnt2)
        tot = tot1 + tot2
        return (measure(cnt1) * tot1 + measure(cnt2) * tot2) / tot

    def __evaluate_split(self, df, class_col, split_col, feature_val, measure):
        df1, df2 = df[df[split_col] <= feature_val], df[df[split_col] > feature_val]
        return self.__wavg(df1[class_col], df2[class_col], measure)

    def __best_split_for_column(self, df, class_col, split_col, method):
        best_v = ''
        best_mae = float("inf")

        for v in set(df[split_col]):

            mae = self.__evaluate_split(df, class_col, split_col, v, method)
            if mae < best_mae:
                best_v = v
                best_mae = mae

        return best_v, best_mae

    def __best_split(self, df, class_col, method):
        best_col = 0
        best_v = ''
        best_mae = float("inf")

        for split_col in df.columns:
            if split_col != class_col:
                v, mae = self.__best_split_for_column(df, class_col, split_col, method)
                if mae < best_mae:
                    best_v = v
                    best_mae = mae
                    best_col = split_col

        return best_col, best_v, best_mae

    def __predict_row(self, model, row):
        if model[0] is None:
            return model[3]
        if row[model[0]] <= model[1]:
            return self.__predict_row(model[6], row)
        else:
            return self.__predict_row(model[7], row)


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
        return 1 / (1 + np.exp(-X @ self.weights))

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

    def train(self, X, y):
        self.X = X
        self.y = y
        for _ in range(1000):
            o = self._feed_forward(X)
            self._backward(X, y, o)

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(self.X))
        print("Hidden Layer: \n" + str(self.weights1))
        print("Output: \n" + str(self._feed_forward(self.X)))
        return self._feed_forward(self.X)

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

    def _backward(self, X, y, o):
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

        # Adjustment to first set of weights (input => hidden)
        self.weights1 += X.T.dot(self.z2_delta)

        # Adjustment to second set of weights (hidden => output)
        self.weights2 += self.activated_hidden.T.dot(self.o_delta)


class AutoencoderTF(Model):
    def __init__(self, hidden_dim=3, encoding_dim=8):
        super(AutoencoderTF, self).__init__()
        self.hidden_layer = layers.Dense(hidden_dim)
        self.output_layer = layers.Dense(encoding_dim, activation='sigmoid')

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)

    def train(self, X, y, epochs=10000, learning_rate=1e-3):
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=losses.MeanSquaredError())
        self.fit(X, y, epochs=epochs, shuffle=True, verbose=0)

    def test(self, y):
        print("Input Data: " + str(y) + "\n" + "Predicted data based on trained weights: " + str(self.predict(y)))
        print("Weights: " + str(self.weights))
        return self.predict(y)


class MultiNeuralNetwork:
    def __init__(self, input_size=13, hidden_size=8, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        self.X = X
        self.y = y
        for i in range(epochs):
            self._forward(X)
            self._backward(X, y, learning_rate)

    def predict(self, X):
        self._forward(X)
        print("Predicted data based on trained weights: " + str(self.yHat))
        print("Weights: " + str(self.W1) + "\n" + str(self.W2))
        print("Input (scaled): \n" + str(self.X))
        print("Hidden Layer: \n" + str(self.z2))
        print("Output: \n" + str(self.yHat))
        return self.yHat

    def _forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = _sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = _sigmoid(self.z3)

    def _backward(self, X, y, learning_rate):
        self.yHat_error = y - self.yHat
        self.yHat_delta = self.yHat_error * _sigmoidPrime(self.yHat)

        self.a2_error = self.yHat_delta.dot(self.W2.T)
        self.a2_delta = self.a2_error * _sigmoidPrime(self.a2)

        # Update weights
        self.W1 += X.T.dot(self.a2_delta) * learning_rate
        self.W2 += self.a2.T.dot(self.yHat_delta) * learning_rate


def normalize(data):
    avg = data.mean()
    stdev = data.std()
    return (data - avg) / stdev, avg, stdev


# Entropy Method
# Entropy is a measure of information that indicates the disorder of the features with the target.
# Similar to the Gini Index, the optimum split is chosen by the feature with less entropy.
def entropy(labels):
    cnt = Counter(labels)
    tot = sum(cnt.values())
    return sum([-cnt[i] / tot * log2(cnt[i] / tot) for i in cnt])


# Variance
def variance(labels):
    return np.var(labels)


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
    # %%
