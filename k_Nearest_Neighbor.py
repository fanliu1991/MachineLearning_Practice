'''
pros: high accuracy, robust to outliers, no assumption for input data
cons: high time complexity, high space complexity
'''

from numpy import *
from sklearn.neighbors import KNeighborsClassifier
import operator
import matplotlib.pyplot as plt

# ceate dataset
def file2matrix(filename):
    fr = open(filename)
    lines_array = fr.readlines()
    lines_num = len(lines_array)

    attr_num = 3 ''' hard code'''
    data_matrix = numpy.zeros((lines_num, attr_num))

    data_label = []
    index = 0
    for line in lines_array:
        line = line.strip()
        line_content = line.split(",")
        data_matrix[index, :] = line_content[0:attr_num]
        data_label.append(int(line_content[-1]))
        index += 1

    return data_matrix, data_label

data, labels = file2matrix("dataset_example.txt")
fig = plt.figure()
ax = fig.subplot(nrows=1, ncols=1, index=1)
ax.scatter(data[:, 1], data[:, 2], 15.0*array(labels), 15.0*array(labels))
plt.show()

# normalization
norm_data = (data - data.min(0)) / (data.max(0) - data.min(0))

# split training dataset and test dataset
ratio = 0.9
split_num = int(norm_data.shape[0] * ratio)
training_data = norm_data[:split_num, :]
training_labels = labels[:split_num]

test_data = norm_data[split_num:, :]
test_labels = str(labels[split_num:])


# KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(training_data, training_labels)

predict_labels = str(neigh.predict(test_data))
error_count = 0

for i in xrange(len(test_labels)):
    if predict_labels[i] != test_labels:
        error_count += 1

accuracy = error_count * 1.0 / len(test_labels)
print "***** ", accuracy, " *****"
