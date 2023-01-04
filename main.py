import sys, csv, numpy as np
import helper_functions as helper
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier
import matplotlib.pyplot as plt



raw_data_set = []
label_array = []

with open('data.csv') as data_file:
    reader = csv.reader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        raw_data_set.append(row);


with open('names.csv') as data_file:
    reader = csv.reader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        label_array.append(row);
        
raw_data_set = list(zip(*raw_data_set))

label_encoder = LabelEncoder()

for i in range(len(raw_data_set)):
    if type(raw_data_set[i][0]) != float:
        raw_data_set[i] = label_encoder.fit_transform(raw_data_set[i])
        
data_set = np.array(raw_data_set)
data_set = data_set.T.astype(float)

train_data = data_set[:,:20]
class_data = data_set[:,-1]

knn = helper.train_knn(train_data, class_data, 5)

rand_data = helper.generate_random_row(train_data)

result = helper.predict_random_new_data(train_data, label_array, knn)

#helper.run_kfold(knn, train_data, class_data, 5)
