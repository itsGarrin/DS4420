import multiprocessing as mp
from statistics import mean, mode

import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, losses
from keras.models import Model
from scipy import sparse
from scipy.stats import multivariate_normal, binom
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle, resample


class DecisionTree:
    def __init__(self, criterion="entropy", regression=False, max_depth=10, min_instances=1, target_impurity=0.01):
        self.criterion = criterion
        self.regression = regression
        self.max_depth = max_depth
        self.min_instances = min_instances
        self.target_impurity = target_impurity
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.tree = self.__build_tree(X, y)

    def __build_tree(self, X: pd.DataFrame, y: pd.DataFrame, depth=0):
        if self.regression:
            return self.__build_tree_regression(X, y, depth)
        else:
            return self.__build_tree_classification(X, y, depth)

    def __build_tree_regression(self, X: pd.DataFrame, y: pd.DataFrame, depth: int):
        if depth == self.max_depth:
            return self.__leaf_regression(y)
        elif len(y) <= self.min_instances:
            return self.__leaf_regression(y)
        elif self.__impurity(y) <= self.target_impurity:
            return self.__leaf_regression(y)
        else:
            depth += 1
            best_split = self.__best_split(X, y)
            left_X = X[X[best_split[0]] <= best_split[1]]
            left_y = y[X[best_split[0]] <= best_split[1]]
            right_X = X[X[best_split[0]] > best_split[1]]
            right_y = y[X[best_split[0]] > best_split[1]]
            return [best_split[0], best_split[1], self.__build_tree_regression(left_X, left_y, depth),
                    self.__build_tree_regression(right_X, right_y, depth)]

    def __build_tree_classification(self, X: pd.DataFrame, y: pd.DataFrame, depth: int):
        if depth == self.max_depth:
            return self.__leaf_classification(y)
        elif len(y) <= self.min_instances:
            return self.__leaf_classification(y)
        elif self.__impurity(y) <= self.target_impurity:
            return self.__leaf_classification(y)
        else:
            depth += 1
            best_split = self.__best_split(X, y)
            left_X = X[X[best_split[0]] <= best_split[1]]
            left_y = y[X[best_split[0]] <= best_split[1]]
            right_X = X[X[best_split[0]] > best_split[1]]
            right_y = y[X[best_split[0]] > best_split[1]]
            return [best_split[0], best_split[1], self.__build_tree_classification(left_X, left_y, depth),
                    self.__build_tree_classification(right_X, right_y, depth)]

    def __leaf_regression(self, y: pd.DataFrame):
        return y.mean()

    def __leaf_classification(self, y: pd.DataFrame):
        return mode(y)

    def __best_split(self, X: pd.DataFrame, y: pd.DataFrame):
        best_feature = None
        best_value = None
        best_score = np.inf

        for feature in X.columns:
            for value in X[feature]:
                score = self.__score(X, y, feature, value)
                if score < best_score:
                    best_feature = feature
                    best_value = value
                    best_score = score

        return best_feature, best_value

    def __score(self, X: pd.DataFrame, y: pd.DataFrame, feature: str, value: float):
        left = y[X[feature] <= value]
        right = y[X[feature] > value]

        return self.__impurity(left) * len(left) + self.__impurity(right) * len(right)

    def __impurity(self, y: pd.DataFrame):
        if self.criterion == "entropy":
            return self.__entropy(y)
        elif self.criterion == "gini":
            return self.__gini(y)
        elif self.criterion == "mse":
            return self.__mse(y)
        else:
            raise ValueError("Invalid criterion")

    def __entropy(self, y: pd.DataFrame):
        if len(y) == 0:
            return 0
        else:
            p = y.value_counts() / len(y)
            return -sum(p * np.log2(p))

    def __gini(self, y: pd.DataFrame):
        if len(y) == 0:
            return 0
        else:
            p = y.value_counts() / len(y)
            return 1 - sum(p ** 2)

    def __mse(self, y: pd.DataFrame):
        if len(y) == 0:
            return 0
        else:
            return np.var(y)

    def predict(self, X: pd.DataFrame):
        return X.apply(self.__predict_row, axis=1)

    def __predict_row(self, row: pd.Series):
        return self.__predict_row_helper(row, self.tree)

    def __predict_row_helper(self, row: pd.Series, tree):
        if isinstance(tree, float) or isinstance(tree, int):
            return tree
        else:
            if row[tree[0]] <= tree[1]:
                return self.__predict_row_helper(row, tree[2])
            else:
                return self.__predict_row_helper(row, tree[3])


class LinearRegression:
    def __init__(self):
        self.w = None
        return

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]

        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.w)


class RidgeLinearRegression:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.weights = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X.T @ X + self.alpha * np.identity(X.shape[1])) @ X.T @ y

    def predict(self, X: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights


class LinearGradientDescent:
    def __init__(self, alpha=1, learning_rate=1e-4, iterations=100):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            predictions = np.dot(X, self.weights)
            gradient = (X.T.dot(predictions - y) + self.alpha * self.weights)
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights


class LogisticGradientDescent:
    def __init__(self, alpha=1, learning_rate=1e-4, iterations=100):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.zeros(X.shape[1])

        for i in range(self.iterations):
            predictions = _sigmoid(X @ self.weights)
            gradient = (X.T @ (predictions - y) + self.alpha * self.weights)
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return _sigmoid(X @ self.weights)


class Perceptron:
    def __init__(self, learning_rate=5e-2):
        self.learning_rate = learning_rate
        self.weights = None
        self.mistakes = None

    def fit(self, X: pd.DataFrame):
        X = np.c_[np.ones(X.shape[0]), X]
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

    def predict(self, X: pd.DataFrame):
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


class GaussianDiscriminantAnalysis:
    def __init__(self, epsilon=1e-9):
        self.mean = None
        self.covariance = None
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.mean = X.groupby(y).mean()
        self.covariance = X.groupby(y).cov() + self.epsilon

    def predict(self, X: pd.DataFrame):
        return self.__predict(X).idxmax(axis=1)

    def __predict(self, X: pd.DataFrame):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return pd.DataFrame(predictions)

    def __predict_row(self, row: pd.Series):
        probabilities = []
        for label in self.mean.index:
            probabilities.append(self.__gaussian(row, self.mean.loc[label], self.covariance.loc[label]))
        return pd.Series(probabilities, index=self.mean.index)

    def __gaussian(self, x: pd.Series, mean: pd.Series, covariance: pd.DataFrame):
        return (1 / np.sqrt(np.linalg.det(covariance))) * np.exp(
            -0.5 * (x - mean).T @ np.linalg.inv(covariance) @ (x - mean))


class BernoulliNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.priors = None
        self.threshold = None
        self.probs = {}
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.priors = X.groupby(y).size() / len(X)
        self.threshold = X.mean().mean()

        X = X >= self.threshold

        for label in y.unique():
            samples = X[y == label]
            self.probs[label] = samples.mean() + self.epsilon

    def predict(self, X: pd.DataFrame):
        return pd.DataFrame(self.__predict(X))

    def __predict(self, X: pd.DataFrame):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return predictions

    def __predict_row(self, row: pd.Series):
        probabilities = []
        for label in self.probs:
            probabilities.append(self.__bernoulli(row, self.threshold, self.probs[label], self.priors[label]))
        return pd.Series(probabilities, index=self.probs.keys())

    def __bernoulli(self, x: pd.Series, threshold: float, prob: float, prior: float):
        return np.sum(np.where(x >= threshold, np.log(prob), np.log(1 - prob))) + np.log(prior)


class GaussianNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.priors = None
        self.means = None
        self.variances = None
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.priors = X.groupby(y).size() / len(X)
        self.means = X.groupby(y).mean()
        self.variances = X.groupby(y).var() + self.epsilon

    def predict(self, X: pd.DataFrame):
        return self.__predict(X)

    def __predict(self, X: pd.DataFrame):
        predictions = []
        for row in X.iterrows():
            row = row[1]
            predictions.append(self.__predict_row(row))
        return pd.DataFrame(predictions)

    def __predict_row(self, row: pd.Series):
        probabilities = []
        for label in self.means.index:
            probabilities.append(
                self.__gaussian(row, self.means.loc[label], self.variances.loc[label], self.priors[label]))
        return pd.Series(probabilities, index=self.means.index)

    def __gaussian(self, x: pd.Series, mean: pd.Series, variance: pd.DataFrame, prior: float):
        return np.log(prior) + np.sum(-0.5 * np.log(2 * np.pi * variance) - 0.5 * ((x - mean) ** 2 / variance))


class GaussianEM:
    def __init__(self, dim: int, k: int, epochs=500, epsilon=1e-9):
        self.means = [np.random.normal(size=dim) for _ in range(k)]
        self.covariances = [np.eye(dim) for _ in range(k)]
        self.weights = np.ones(k) / k
        self.dim = dim
        self.k = k
        self.epsilon = epsilon
        self.epochs = epochs
        self.memberships = None

    def fit(self, X: pd.DataFrame):
        for _ in range(self.epochs):
            self.__e_step(X)
            self.__m_step(X)

        return self.means, self.covariances

    def __e_step(self, X: pd.DataFrame):
        self.memberships = np.zeros((X.shape[0], self.weights.shape[0]))
        for i in range(self.weights.shape[0]):
            self.memberships[:, i] = self.weights[i] * multivariate_normal.pdf(X, self.means[i], self.covariances[i])

        self.memberships = self.memberships / self.memberships.sum(axis=1, keepdims=True)

    def __m_step(self, X: pd.DataFrame):
        self.weights = self.memberships.sum(axis=0) / X.shape[0]

        for i in range(self.weights.shape[0]):
            self.means[i] = np.sum(self.memberships[:, i, None] * X, axis=0) / self.memberships[:, i].sum()

            diff = X - self.means[i]
            self.covariances[i] = np.dot(self.memberships[:, i] * diff.T, diff) / self.memberships[:,
                                                                                  i].sum() + self.epsilon


class BinomialEM:
    def __init__(self, k: int, epochs=500, epsilon=1e-9):
        self.probs = np.random.random((k, 1))
        self.weights = np.ones(k) / k
        self.k = k
        self.epsilon = epsilon
        self.epochs = epochs
        self.memberships = None

    def fit(self, X: pd.DataFrame):
        for _ in range(self.epochs):
            self.__e_step(X)
            self.__m_step(X)

        return self.weights, self.probs

    def __e_step(self, X: pd.DataFrame):
        self.memberships = np.zeros((X.shape[0], self.weights.shape[0]))
        for i in range(self.weights.shape[0]):
            self.memberships[:, i] = self.weights[i] * np.prod(binom.pmf(X, 1, self.probs[i]), axis=1)

        self.memberships = self.memberships / self.memberships.sum(axis=1, keepdims=True)

    def __m_step(self, X: pd.DataFrame):
        self.weights = self.memberships.sum(axis=0) / X.shape[0]

        for i in range(self.weights.shape[0]):
            self.probs[i] = np.sum(self.memberships[:, i][:, None] * X, axis=0).sum() / (
                    self.memberships[:, i].sum() * X.shape[1])


class AdaBoost:
    def __init__(self, num_classifiers=80, learning_rate=0.5, splitter="best"):
        self.num_classifiers = num_classifiers
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []
        self.splitter = splitter
        self.feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y = np.where(y <= 0, -1, 1)
        weights = np.ones(len(X)) / len(X)

        for _ in range(self.num_classifiers):
            stump = DecisionTreeClassifier(max_depth=1, splitter=self.splitter)
            stump.fit(X, y, sample_weight=weights)
            predictions = stump.predict(X)
            error = np.sum(weights[predictions != y]) / np.sum(weights)

            alpha = 1
            if error != 0:
                alpha = self.learning_rate * np.log((1 - error) / error)

            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)

            self.alphas.append(alpha)
            self.models.append(stump)

    def predict(self, X: pd.DataFrame):
        predictions = np.zeros(len(X))

        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X)

        return np.sign(predictions)

    def active_learning(self, X: pd.DataFrame, y: pd.DataFrame, initial_train_size=0.05, step_size=0.025,
                        final_train_size=0.6):
        X, y = shuffle(X, y)
        train_data_features, train_data_labels, rest_data_features, rest_data_labels = train_test_split(X, y,
                                                                                                        train_size=initial_train_size)

        self.fit(train_data_features, train_data_labels)
        y_pred = self.predict(train_data_features)
        accuracy_scores = [accuracy(train_data_labels, y_pred)]

        while len(train_data_features) < len(X) * final_train_size:
            # computing the absolute prediction scores
            rest_data_scores = self.predict(rest_data_features)

            # selecting the samples closest to decision surface
            closest_samples_indices = np.argsort(np.abs(rest_data_scores))
            closest_samples_indices = closest_samples_indices[:int(step_size * len(X))]

            # adding the selected samples to the training data
            train_data_features = pd.concat([train_data_features, rest_data_features.iloc[closest_samples_indices]])
            train_data_labels = pd.concat([train_data_labels, rest_data_labels.iloc[closest_samples_indices]])

            # removing the selected samples from the remaining data
            rest_data_features = rest_data_features.drop(rest_data_features.index[closest_samples_indices])
            rest_data_labels = rest_data_labels.drop(rest_data_labels.index[closest_samples_indices])

            self.fit(train_data_features, train_data_labels)
            y_pred = self.predict(train_data_features)
            accuracy_scores.append(accuracy(train_data_labels, y_pred))

        return mean(accuracy_scores)

    def __get_feature_importance(self):
        feature_scores = {}
        coef_sum = 0
        for alpha, stump in zip(self.alphas, self.models):
            coef_sum += alpha
            ind = list(stump.feature_importances_).index(max(stump.feature_importances_))
            prev_val = feature_scores.get(ind, 0)
            feature_scores[ind] = prev_val + alpha / coef_sum

        # convert to DataFrame for easy sorting and plotting
        feature_df = pd.DataFrame(list(feature_scores.items()), columns=['Feature', 'Importance'])
        feature_df = feature_df.sort_values('Importance', ascending=False)

        return feature_df

    def get_top_features(self, n):
        feature_df = self.__get_feature_importance()
        return feature_df.iloc[:n, :]


class ECOC:
    def __init__(self, ecoc_codes: np.ndarray):
        self.ecoc_codes = ecoc_codes
        self.estimators = []
        self.classes = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.classes = np.unique(y)

        for column_index in range(self.ecoc_codes.shape[1]):
            binary_y = self.__encode(y, column_index)

            estimator = AdaBoost(num_classifiers=200)
            estimator.fit(pd.concat([X, pd.DataFrame(binary_y)], axis=1))
            self.estimators.append(estimator)

    def __encode(self, y, column_index):
        binary_y = np.zeros(len(y))
        for label in self.classes:
            binary_bit = self.ecoc_codes[self.classes == label, column_index]
            binary_y[y == label] = binary_bit
        return binary_y

    def predict(self, test):
        X = test.iloc[:, :-1]
        predictions = np.zeros((len(X), len(self.estimators)))
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(test)
        return self.__decode(predictions)

    def __decode(self, predictions):
        decoded = []
        for row in predictions:
            decoded.append(self.__decode_row(row))
        return decoded

    def __decode_row(self, row):
        distances = []
        for code in self.ecoc_codes:
            distances.append(np.sum(code != row))
        return self.classes[np.argmin(distances)]


class Bagging:
    def __init__(self, n_estimators=50, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        for _ in range(self.n_estimators):
            X_resample, y_resample = resample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_resample, y_resample)
            self.trees.append(tree)

    def predict(self, test):
        X = test.iloc[:, :-1]
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return [mode(pred) for pred in np.transpose(predictions)]


class GradientBoosting:
    def __init__(self, n_estimators=10, max_depth=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        self.trees = []
        Yx = np.array(y)
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, Yx)
            self.trees.append(tree)
            Yx = Yx - tree.predict(X)

    def predict(self, test):
        X = test.iloc[:, :-1]
        return sum(tree.predict(X) for tree in self.trees)

    def mse(self, test):
        y_pred = self.predict(test)
        y = test.iloc[:, -1]
        return np.mean((y_pred - y) ** 2)


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


class MissingValuesBernoulliNaiveBayes:
    def __init__(self, epsilon=1e-9):
        self.priors = None
        self.threshold = None
        self.probs = {}
        self.epsilon = epsilon

    def fit(self, train):
        X = train.iloc[:, :-1]
        y = train.iloc[:, -1]
        self.priors = X.groupby(y).size() / len(train)
        self.threshold = X.apply(np.nanmean, axis=0)

        X = X >= self.threshold

        for label in y.unique():
            samples = X[y == label]
            self.probs[label] = (samples >= self.threshold).mean() + self.epsilon

    def predict(self, test):
        X = test.iloc[:, :-1]
        X = X.apply(lambda column: column.fillna(column.mean()), axis=0)
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
        factors = np.where(x.notna(), np.where(x >= threshold, np.log(prob), np.log(1 - prob)), 0)
        return np.sum(factors) + np.log(prior)

    def accuracy(self, test):
        y = test.iloc[:, -1]
        predictions = self.predict(test).idxmax(axis=1)
        return np.sum(predictions == y) / len(y)


class SVM:
    def __init__(self, C=1.0, tol=1e-5, max_passes=10, iterations=1000):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.iterations = iterations
        self.X = None
        self.y = None
        self.alpha = None
        self.b = None
        self.threshold = 1

    def fit(self, train):
        X = train.iloc[:, :-1].to_numpy()
        y = train.iloc[:, -1]
        y = np.where(y == self.threshold, 1, -1)
        self.X = X
        self.y = y
        m, n = X.shape
        alpha = np.zeros(m)
        b = 0
        # passes = 0

        # compute gram matrix
        K = np.dot(X, X.T)

        for _ in range(self.iterations):
            # while passes < self.max_passes:
            # num_changed_alphas = 0

            for i in range(m):
                Ei = np.dot(K[i], alpha * y) - y[i] + b

                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = np.random.choice(np.delete(np.arange(m), i))
                    Ej = np.dot(K[j], alpha * y) - y[j] + b

                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]

                    if eta >= 0:
                        continue

                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)

                    if np.abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    alpha[i] += y[j] * y[i] * (alpha_j_old - alpha[j])

                    b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    break

        self.alpha = alpha
        self.b = b

    def predict(self, test):
        X = test.iloc[:, :-1].to_numpy()

        y_pred = []
        for x in X:
            prediction = np.sign(np.dot(self.alpha * self.y, np.dot(self.X, x)) + self.b)
            y_pred.append(prediction)

        return np.array(y_pred)

    def cross_validate(self, train, k=10):
        fold_size = int(len(train) / k)
        results = []

        # Prepare the input for the workers
        for i in range(k):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            results.append((training_fold.copy(), validation_fold.copy()))

        # Create a pool of workers and pass them the work
        with mp.Pool(mp.cpu_count()) as pool:
            accuracies = pool.map(self.evaluate_fold, results)

        return np.mean(accuracies)

    def evaluate_fold(self, folds_data):
        training_fold, validation_fold = folds_data
        self.fit(training_fold)
        predictions = self.predict(validation_fold)
        labels = validation_fold[validation_fold.columns[-1]]
        labels = np.where(labels == self.threshold, 1, -1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return accuracy

    def ovr(self, train, classes=10):
        fold_size = int(len(train) / classes)
        accuracies = []

        # Prepare the input for the workers
        for i in range(classes):
            # Shuffle data
            train = train.sample(frac=1).reset_index(drop=True)
            training_fold = pd.concat([train.iloc[:i * fold_size], train.iloc[(i + 1) * fold_size:]]).reset_index(
                drop=True)
            validation_fold = train.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

            self.threshold = i
            self.fit(training_fold)
            predictions = self.predict(validation_fold)

            labels = validation_fold.iloc[:, -1]
            labels = np.where(labels == self.threshold, 1, -1)

            accuracy = np.sum(predictions == labels) / len(labels)
            accuracies.append(accuracy)
            print('Class ' + str(i) + ' Accuracy: ' + str(accuracy))

        return np.mean(accuracies)


class KNN:
    def __init__(self, k=1, bandwidth=1.1):
        self.k = k
        self.bandwidth = bandwidth
        self.X = None
        self.y = None

    def fit(self, train):
        self.X = train.iloc[:, :-1]
        self.y = train.iloc[:, -1]

    def predict(self, test):
        X = test.iloc[:, :-1].to_numpy()

        predictions = []
        for x in X:
            predictions.append(self.__predict_row(x))

        return np.array(predictions)

    def __predict_row(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        k_nearest_labels = self.y.iloc[np.argsort(distances)[:self.k]]
        return self.__gaussian(x, k_nearest_labels)

    def __gaussian(self, x, k_nearest_labels):
        # Calculate the distances
        distances = np.linalg.norm(x - self.X.iloc[k_nearest_labels], axis=1)

        # Calculate the numerator and denominator separately
        numerator = np.sum(k_nearest_labels * np.exp(-distances ** 2 / (2 * self.bandwidth ** 2)))
        denominator = np.sum(np.exp(-distances ** 2 / (2 * self.bandwidth ** 2)))

        # Return the division result
        return numerator / denominator

    def accuracy(self, test):
        y = test.iloc[:, -1]
        predictions = self.predict(test)
        return np.sum(predictions == y) / len(y)


def mse(y: pd.DataFrame, y_pred: pd.DataFrame):
    return np.mean((y_pred - y) ** 2)


def accuracy(y: pd.DataFrame, y_pred: pd.DataFrame):
    return np.sum(y_pred == y) / len(y)


def cross_validate(model: object, X: pd.DataFrame, y: pd.DataFrame, k=10, confusion=False, idxmax=False):
    fold_size = int(len(X) / k)
    accuracies = []
    matrices = np.zeros(shape=(2, 2), dtype=int)

    for i in range(k):
        # Shuffle data
        X, y = shuffle(X, y)
        training_fold = pd.concat([X.iloc[:i * fold_size], X.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
        validation_fold = X.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)
        training_labels = pd.concat([y.iloc[:i * fold_size], y.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
        validation_labels = y.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

        model.fit(training_fold, training_labels)
        predictions = model.predict(validation_fold)

        if idxmax:
            predictions = predictions.idxmax(axis=1)

        predictions = np.where(predictions >= 0.5, 1, 0)

        matrices += confusion_matrix(validation_labels, predictions)

        accuracy = np.sum(predictions == validation_labels) / len(validation_labels)
        accuracies.append(accuracy)

    if confusion:
        return np.mean(accuracies), matrices
    else:
        return np.mean(accuracies)


def multiprocess_cv(model: object, X: pd.DataFrame, y: pd.DataFrame, k=10):
    fold_size = int(len(X) / k)
    results = []

    # Prepare the input for the workers
    for i in range(k):
        # Shuffle data
        X, y = shuffle(X, y)
        training_fold = pd.concat([X.iloc[:i * fold_size], X.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
        validation_fold = X.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)
        training_labels = pd.concat([y.iloc[:i * fold_size], y.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
        validation_labels = y.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

        results.append((model, training_fold, validation_fold, training_labels, validation_labels))

    # Create a pool of workers and pass them the work
    with mp.Pool(mp.cpu_count()) as pool:
        accuracies = pool.map(evaluate_fold, results)

    return np.mean(accuracies)


def evaluate_fold(folds_data: tuple):
    model, training_fold, validation_fold, training_labels, validation_labels = folds_data
    model.fit(training_fold, training_labels)
    predictions = model.predict(validation_fold)
    predictions = np.where(predictions >= 0.5, 1, 0)
    accuracy = np.sum(predictions == validation_labels) / len(validation_labels)

    return accuracy


def roc_curve(model: object, X: pd.DataFrame, y: pd.DataFrame, num_thresholds=51, k=10, nb=False):
    fold_size = int(len(X) / k)

    i = k - 1

    # Shuffle data
    X, y = shuffle(X, y)
    training_fold = pd.concat([X.iloc[:i * fold_size], X.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
    validation_fold = X.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)
    training_labels = pd.concat([y.iloc[:i * fold_size], y.iloc[(i + 1) * fold_size:]]).reset_index(drop=True)
    validation_labels = y.iloc[i * fold_size:(i + 1) * fold_size].reset_index(drop=True)

    model.fit(training_fold, training_labels)
    pred = model.predict(validation_fold)

    thresholds = np.linspace(np.min(pred), np.max(pred), num_thresholds)

    if nb:
        pred = pred.iloc[:, 1] - pred.iloc[:, 0]
        thresholds = np.unique(pred)
        thresholds = np.sort(thresholds)[::-1]

    matrices = np.zeros(shape=(len(thresholds), 2, 2), dtype=int)

    threshold_matrices = []
    for threshold in thresholds:
        vector = np.vectorize(lambda y: 1 if y >= threshold else 0)
        predictions = vector(pred)

        threshold_matrices.append(confusion_matrix(validation_labels, predictions))

    # sum matrices along axis
    matrices = np.add(matrices, threshold_matrices)

    return generate_roc_curve(matrices)


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


def load_newsgroup(file_path):
    with open(file_path, 'r') as f:
        data = []
        row = []
        col = []
        labels = []
        i = 0
        for line in f:
            # split the line into label and sparse vector
            parts = line.strip().split()
            label = int(parts[0])
            labels.append(label)
            # parse the sparse vector
            for pair in parts[1:]:
                idx, val = pair.split(':')
                row.append(i)
                col.append(int(idx))
                data.append(float(val))
            i += 1
        # create the sparse matrix in COO format
        X_coo = sparse.coo_matrix((data, (row, col)))
        # convert COO to CSR format
        X_features = X_coo.tocsr()

        labels = np.array(labels).reshape(-1, 1)
        labels = sparse.csr_matrix(labels)

        # combine the labels and features into a single array
        X = sparse.hstack([X_features, labels])

        df = pd.DataFrame.sparse.from_spmatrix(X)

        return df


def _sigmoid(s):
    return 1 / (1 + np.exp(-s))


def _sigmoidPrime(s):
    return s * (1 - s)
