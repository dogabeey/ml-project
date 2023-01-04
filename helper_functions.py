from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import sys, csv, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def run_kfold(clf, X, y, folds):
    kf = KFold(n_splits=folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome))
    
    
def train_knn(x_train, y_train, neighbors):
  # Create a KNN classifier with X amont neighbors
  knn = KNNClassifier(n_neighbors=3)
  # Fit the classifier to the training data
  knn.fit(x_train, y_train)
  # Return the trained classifier
  return knn


def generate_random_row(data):
  # Get the minimum and maximum values of each column
  min_values = np.min(data, axis=0)
  max_values = np.max(data, axis=0)
  
  # Generate random values between the min and max values
  row = np.random.uniform(min_values, max_values)
  
  return row 

def predict_random_new_data(train_data, label_array, classifier):
    rand_data = generate_random_row(train_data)
    result = classifier.predict(rand_data.reshape(1, -1))
    merged_data = np.concatenate((rand_data, result))
    text_merged_data = np.concatenate((merged_data, label_array))
    
    for point in merged_data:
        print((str(round(point,2)) + "\t").expandtabs(15), end = ' ')
    for point in label_array:
        print((str(point) + "\t").expandtabs(15), end = ' ')