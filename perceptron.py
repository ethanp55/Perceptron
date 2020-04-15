import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle if deterministic is not None else False
        self.deterministic = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # Initialize the weights
        self.num_features = np.shape(X)[1]
        self.weights = self.initialize_weights() if initial_weights is None else initial_weights

        # Get the number of data instances
        num_instances = np.shape(X)[0]

        # Use numpy to add a bias column (axis=1) to the input
        X = np.concatenate((X, np.ones((num_instances, 1))), axis=1)

        # Set stop flag for our training loop and initialize the number of epochs we will run
        stop = False
        self.n_epochs = 0

        # A list we will use to store the last 5 accuracies in
        # j will be used to index into the list of accuracies
        accuracies = [0] * 5
        j = 0

        # List of previous misclassification rates and the average misclassification rate (used for Part 4)
        self.misclassification_rates = []

        # Run through epochs until our stopping criteria is met
        while not stop:
            # Shuffle the data at the start of each epoch (if we are not running deterministically)
            if self.shuffle:
                X, y = self._shuffle_data(X, y)

            for i in range(np.shape(X)[0]):
                # Use our predict method
                output = self.predict([X[i]])

                # Update the weights
                weight_updates = self.lr * (y[i] - output) * X[i]
                self.weights += weight_updates

            # Calculate the accuracy at the end of the epoch
            accuracy = self.score(X, y)

            # Update the list of accuracies
            accuracies[j] = accuracy

            # Update j (it will wrap around back to 0 if it is too big)
            j = j + 1 if j < 4 else 0

            # Append the misclassification rate to the history of misclassifications (used for analysis in Part 4)
            self.misclassification_rates.append(1 - accuracy)

            # Increment the number of epochs
            self.n_epochs += 1

            # If we are running deterministically, check if we've reached the set number of epochs
            if self.deterministic is not None and self.n_epochs == self.deterministic:
                break

            # If we are not running deterministically, check if the stopping criteria is met
            elif self.n_epochs >= 5 and self.deterministic is None:
                stop = True if abs(max(accuracies) - min(accuracies)) < 0.05 else False

            # If we ever run too many epochs, just stop
            if self.n_epochs >= 1000:
                stop = True

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        # If there is no bias feature, add it
        if np.shape(X)[1] != self.num_features + 1:
            X = np.concatenate((X, np.ones((np.shape(X)[0], 1))), axis=1)

        # Use numpy to calculate the predicted output values
        net = np.dot(X, np.transpose(self.weights))

        # Use numpy to convert the predicted output values to predicted target values
        return np.where(net > 0, 1, 0)

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        # Use num_features + 1 to account for an extra weight for the bias
        return np.zeros((1, self.num_features + 1))

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        # Use our predict method
        output = self.predict(X)

        # Find the total number of correct predictions and the total number of data instances
        num_correct = sum(output == y)[0]
        num_instances = np.shape(X)[0]

        # Calculate and return the accuracy
        return num_correct / num_instances

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        # Use numpy to create a random permutation of the numbers 0 to the number of data instances
        p = np.random.permutation(np.shape(X)[0])

        # Use the permutation to return shuffled X and y data
        # Using the same permutation for both X and y ensures that they are shuffled in unison
        return X[p], y[p]

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        # Returns the weights
        # INCLUDES THE BIAS WEIGHT
        return self.weights
