import sys, csv, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier
import matplotlib.pyplot as plt

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

raw_data_set = []

with open('data.csv') as data_file:
    reader = csv.reader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        raw_data_set.append(row);
        
raw_data_set = list(zip(*raw_data_set))

label_encoder = LabelEncoder()

for i in range(len(raw_data_set)):
    if type(raw_data_set[i][0]) != float:
        raw_data_set[i] = label_encoder.fit_transform(raw_data_set[i])
        
data_set = np.array(raw_data_set)
data_set = data_set.T.astype(float)

knn = train_knn(data_set[:,:20], data_set[:,-1], 5)

rand_data = generate_random_row(data_set[:,:20])

result = knn.predict(rand_data.reshape(1, -1))

