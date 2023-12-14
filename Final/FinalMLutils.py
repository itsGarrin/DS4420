import keras.optimizers
import numpy as np
from keras import layers, losses
from keras.models import Model
from scipy.stats import binom


class GaussianFilter:
    def __init__(self, size, sigma, image_shape):
        self.size = size
        self.sigma = sigma
        self.image_shape = image_shape

    def _create_gaussian_kernel(self):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * self.sigma ** 2)) * np.exp(
                -((x - (self.size - 1) / 2) ** 2 + (y - (self.size - 1) / 2) ** 2) / (2 * self.sigma ** 2)),
            (self.size, self.size)
        )
        return kernel / np.sum(kernel)

    def apply(self, images):
        kernel = self._create_gaussian_kernel()
        results = []
        for image in images:
            image = image.reshape(self.image_shape)
            filtered_image = np.zeros_like(image, dtype=float)

            for i in range(self.size // 2, image.shape[0] - self.size // 2):
                for j in range(self.size // 2, image.shape[1] - self.size // 2):
                    # Extract the neighborhood
                    neighborhood = image[
                                   max(0, i - self.size // 2): min(image.shape[0],
                                                                   i + self.size // 2 + 1),
                                   max(0, j - self.size // 2): min(image.shape[1], j + self.size // 2 + 1)
                                   ]

                    # Apply the gaussian filter
                    filtered_image[i, j] = np.sum(neighborhood * kernel)

            results.append(filtered_image.flatten())

        return results


class MedianFilter:
    def __init__(self, kernel_size, image_shape):
        self.kernel_size = kernel_size
        self.image_shape = image_shape

    def apply(self, images):
        """
        Apply the median filter to the input image.

        Parameters:
        - image: 2D numpy array representing the grayscale image.

        Returns:
        - result: 2D numpy array, the filtered image.
        """
        results = []

        # Iterate over the image
        for image in images:
            image = image.reshape(self.image_shape)
            result = np.zeros_like(image)

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # Extract the neighborhood
                    neighborhood = image[
                                   max(0, i - self.kernel_size // 2): min(image.shape[0],
                                                                          i + self.kernel_size // 2 + 1),
                                   max(0, j - self.kernel_size // 2): min(image.shape[1], j + self.kernel_size // 2 + 1)
                                   ]

                    # Apply the median filter
                    result[i, j] = np.median(neighborhood)

            results.append(result.flatten())

        return results


class AutoencoderTF(Model):
    def __init__(self, hidden_dim=3, encoding_dim=8):
        super(AutoencoderTF, self).__init__()
        self.hidden_layer = layers.Dense(hidden_dim)
        self.output_layer = layers.Dense(encoding_dim, activation='sigmoid')

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)

    def train(self, X, y, epochs=10000, learning_rate=1e-3):
        self.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                     loss=losses.MeanSquaredError())
        self.fit(X, y, epochs=epochs, shuffle=True, verbose=1)

    def test(self, y):
        predictions = self.predict(y)
        print("Input Data: " + str(y) + "\n" + "Predicted data based on trained weights: " + str(predictions))
        print("Weights: " + str(self.weights))
        return predictions


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
