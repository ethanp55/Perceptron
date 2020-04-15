from perceptron import *
from arff import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

# Custom function for splitting the data into train and test sets
def split_train_test(data, test_ratio):
    # Calculate the size of the test set
    test_size = int(np.shape(data)[0] * test_ratio)

    # Create a random permutation of the indices in the dataset
    shuffled_indices = np.random.permutation(np.shape(data)[0])

    # Get the indices for the train and test sets
    train_indices = shuffled_indices[test_size:]
    test_indices = shuffled_indices[:test_size]

    # Use the indices create in the last step to index into the data and return the train and test sets
    return data[train_indices], data[test_indices]

# Part 1
# DEBUGGING DATASET RESULTS
mat = Arff("datasets/linsep2nonorigin.arff", label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

PClass = PerceptronClassifier(lr=0.1, shuffle=False, deterministic=10)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)

print("DEBUG DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights()[0])
print()


# EVALUATION DATASET RESULTS
mat = Arff("datasets/data_banknote_authentication.arff",label_count=1)
np_mat = mat.data
data = mat[:,:-1]
labels = mat[:,-1].reshape(-1,1)

P2Class = PerceptronClassifier(lr=0.1,shuffle=False,deterministic=10)
P2Class.fit(data,labels)
Accuracy = P2Class.score(data,labels)

print("EVALUATION DATASET")
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",P2Class.get_weights()[0])
print()


# Part 3
# CUSTOM LINEARLY SEPARABLE DATASET RESULTS
mat = Arff("datasets/custom_lin_sep.arff", label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

PClass = PerceptronClassifier(lr=0.1, shuffle=False)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)

print("CUSTOM LINEARLY SEPARABLE DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights()[0])
print("Number of epochs: " + str(PClass.n_epochs))
print()

# Plot the results
x = np.linspace(-1,1,100)
y = (-PClass.get_weights()[0][0] / PClass.get_weights()[0][1]) * x + PClass.get_weights()[0][2] / PClass.get_weights()[0][1]
X_class1 = data[0:4,0]
Y_class1 = data[0:4,1]
X_class2 = data[4:8,0]
Y_class2 = data[4:8,1]
plt.plot(x, y, '-b')
plt.title("Data and Decision Line For Custom Linearly Separable Dataset")
plt.xlabel("x (feature 1) values")
plt.ylabel("y (feature 2) values")
plt.scatter(X_class1, Y_class1, color="red")
plt.scatter(X_class2, Y_class2, color="green")
line = mpatches.Patch(color="blue", label="Learned Decision Line")
class1 = mpatches.Patch(color="red", label="Class1")
class2 = mpatches.Patch(color="green", label="Class2")
plt.legend(loc="best", handles=[line, class1, class2])
plt.grid()
plt.show()


# CUSTOM NON LINEARLY SEPARABLE DATASET RESULTS
mat = Arff("datasets/custom_non_lin_sep.arff", label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

PClass = PerceptronClassifier(lr=.1, shuffle=False)
PClass.fit(data,labels)
Accuracy = PClass.score(data,labels)

print("CUSTOM NON LINEARLY SEPARABLE DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =",PClass.get_weights()[0])
print("Number of epochs: " + str(PClass.n_epochs))
print()

# Plot the results
x = np.linspace(-1,1,100)
y = (-PClass.get_weights()[0][0] / PClass.get_weights()[0][1]) * x + PClass.get_weights()[0][2] / PClass.get_weights()[0][1]
X_class1 = data[0:4,0]
Y_class1 = data[0:4,1]
X_class2 = data[4:8,0]
Y_class2 = data[4:8,1]
plt.plot(x, y, '-b', label="Learned Decision Line")
plt.title("Data and Decision Line For Custom Non Linearly Separable Dataset")
plt.xlabel("x (feature 1) values")
plt.ylabel("y (feature 2) values")
plt.scatter(X_class1, Y_class1, color="red")
plt.scatter(X_class2, Y_class2, color="green")
line = mpatches.Patch(color="blue", label="Learned Decision Line")
class1 = mpatches.Patch(color="red", label="Class1")
class2 = mpatches.Patch(color="green", label="Class2")
plt.legend(loc="best", handles=[line, class1, class2])
plt.grid()
plt.show()


# Part 4
# VOTING DATASET RESULTS
mat = Arff("datasets/voting.arff", label_count=1)

print("VOTING DATASET")

n_runs = 5
max_epochs = 0

misclassification_rates = []

for i in range(n_runs):
    PClass = PerceptronClassifier(lr=0.1, shuffle=True)

    # Use the custom function for splitting the data into train and test sets
    train, test = split_train_test(mat.data, 0.3)

    # Get the train data, train labels, test data, and test labels
    train_data = train[:,0:-1]
    train_labels = train[:,-1].reshape(-1,1)
    test_data = test[:,0:-1]
    test_labels = test[:,-1].reshape(-1,1)

    # Fit the perceptron on the training data
    PClass.fit(train_data, train_labels)

    # Calculate the train and test accuracies
    train_accuracy = PClass.score(train_data,train_labels)
    test_accuracy = PClass.score(test_data, test_labels)

    # Get the misclassification rates for the current run
    misclassification_rates.append(PClass.misclassification_rates)

    # Print the results
    print("Run: " + str(i + 1))
    print("Number of epochs: " + str(PClass.n_epochs))
    print("Train Accuracy = [{:.2f}]".format(test_accuracy))
    print("Test Accuracy = [{:.2f}]".format(test_accuracy))
    print("Final Weights =",PClass.get_weights()[0])
    print()

    max_epochs = PClass.n_epochs if PClass.n_epochs > max_epochs else max_epochs

# Code for looping through the misclassification rates and getting the average misclassification rate for each epoch
avg_misclassification_rates = []

for j in range(max_epochs):
    total = 0
    n = 0

    for i in range(len(misclassification_rates)):
        if j < len(misclassification_rates[i]):
            total += misclassification_rates[i][j]
            n += 1

    avg_misclassification_rates.append(total / n)

x = range(1,len(avg_misclassification_rates) + 1)
plt.plot(x, avg_misclassification_rates, "-b", label="Average Training Misclassification Rate")
plt.xlabel("Epochs")
plt.ylabel("Average Misclassification Rate")
plt.title("Average Misclassification Rate Across Epochs")
plt.legend(loc="best")
plt.grid()
plt.show()

# Part 5
# SCIKIT-LEARN PERCEPTRON ON VOTING DATASET
from sklearn.linear_model import Perceptron

mat = Arff("datasets/voting.arff", label_count=1)

print("SKLEARN PERCEPTRON")

n_runs = 5

for i in range(n_runs):
    sklearn_perceptron = Perceptron(shuffle=True, tol=0.05, eta0=0.1)

    # Use the custom function for splitting the data into train and test sets
    train, test = split_train_test(mat.data, 0.3)

    # Get the train data, train labels, test data, and test labels
    train_data = train[:,0:-1]
    train_labels = train[:,-1].reshape(-1,1)
    test_data = test[:,0:-1]
    test_labels = test[:,-1].reshape(-1,1)

    # Fit the perceptron on the training data
    sklearn_perceptron.fit(train_data,train_labels)

    # Calculate the train and test accuracies
    train_accuracy = sklearn_perceptron.score(train_data,train_labels)
    test_accuracy = sklearn_perceptron.score(test_data, test_labels)

    # Print the results
    print("Run: " + str(i + 1))
    print("Number of epochs: " + str(sklearn_perceptron.n_iter_))
    print("Train Accuracy = [{:.2f}]".format(test_accuracy))
    print("Test Accuracy = [{:.2f}]".format(test_accuracy))
    print()

# SCIKIT-LEARN PERCEPTRON ON CUSTOM NON LINEARLY SEPARABLE DATASET RESULTS
mat = Arff("datasets/custom_non_lin_sep.arff", label_count=1)
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

sklearn_perceptron = Perceptron(shuffle=False, tol=0.05, eta0=0.1)
sklearn_perceptron.fit(data,labels)
Accuracy = sklearn_perceptron.score(data,labels)

print("CUSTOM NON LINEARLY SEPARABLE DATASET")
print("Accuracy = [{:.2f}]".format(Accuracy))
print("Final Weights =",sklearn_perceptron.coef_[0])
print("Bias Weight =",sklearn_perceptron.intercept_)
print("Number of epochs: " + str(sklearn_perceptron.n_iter_))
print()

# SCIKIT-LEARN PERCEPTRON ON EVALUATION DATASET RESULTS
mat = Arff("datasets/data_banknote_authentication.arff",label_count=1)
np_mat = mat.data
data = mat[:,:-1]
labels = mat[:,-1].reshape(-1,1)

sklearn_perceptron = Perceptron(shuffle=False, max_iter=10, n_iter_no_change=10, eta0=0.1)
sklearn_perceptron.fit(data,labels)
Accuracy = sklearn_perceptron.score(data,labels)

print("EVALUATION DATASET")
print("Accuray = [{:.2f}]".format(Accuracy))
print("Final Weights =",sklearn_perceptron.coef_[0])
print("Bias weight =",sklearn_perceptron.intercept_)
print()
