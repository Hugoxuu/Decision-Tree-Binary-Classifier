import sys
import numpy as np
import csv
import math

# Command line arguments
train_input = sys.argv[1]  # path to the training input .csv file
test_input = sys.argv[2]  # path to the test input .csv file
depth = int(sys.argv[3])  # maximum depth to which the tree should be built
train_out = sys.argv[4]  # path of output .labels file to which the predictions on the training data should be written
test_out = sys.argv[5]  # path of output .labels file to which the predictions on the test data should be written
metric_out = sys.argv[6]  # path of the output .txt file to which metrics such as train and test error should be written


#  data structure for decision tree
class Node(object):
    def __init__(self):
        self.left = None  # left children, split on 0
        self.right = None  # right children, split on 1
        self.val = None  # The attribute is presented by matrix row index value, leaf if val == -1
        self.counter = [0, 0]  # List with size 2 to present #samples split on 0 and 1


# Visualize the decision tree
def printer(node, tab):
    if node.val is not None:
        print(attr_list[node.val], node.counter)
        tab += 1
        if node.left is not None:
            for x in range(tab): print('|  ', end='')
            printer(node.left, tab)
        if node.right is not None:
            for x in range(tab): print('|  ', end='')
            printer(node.right, tab)
    return


# Calculate entropy
def entropy(x):
    percent = sum(x) / len(x)
    if percent == 0 or percent == 1:
        return 0
    else:
        ans = - percent * (math.log2(percent)) - (1 - percent) * (math.log2(1 - percent))
        return ans


# Calculate mutual information I(Y;X) = H(Y) + H(X) − H(Y,X)
def mutualInfo(y, x):
    num = len(x)
    H_Y = entropy(y)
    H_X = entropy(x)
    H_XY = 0
    # calculate the joint entropy
    zero0 = 0
    zero1 = 0
    one0 = 0
    one1 = 0
    for i in range(num):
        if x[i] == 0:
            if y[i] == 0:
                zero0 += 1
            else:
                zero1 += 1
        else:
            if y[i] == 0:
                one0 += 1
            else:
                one1 += 1
    if zero0 != 0:
        H_XY -= (zero0 / num) * math.log2(zero0 / num)
    if zero1 != 0:
        H_XY -= (zero1 / num) * math.log2(zero1 / num)
    if one0 != 0:
        H_XY -= (one0 / num) * math.log2(one0 / num)
    if one1 != 0:
        H_XY -= (one1 / num) * math.log2(one1 / num)
    ans = H_Y + H_X - H_XY  # The final equation
    return ans


# Transfer train csv into (0,1) matrix and get attribute list
def csv2matrix(csv_input, name_as_zero):
    with open(csv_input) as csvfile:
        csv_data = csv.reader(csvfile, delimiter=',')
        num = len(next(csv_data))  # num = (num of attributes) + 1
        D = []
        for row in csv_data:
            new_row = [0] * num
            for x in range(num):
                if row[x] == name_as_zero[x]:
                    new_row[x] = 0
                else:
                    new_row[x] = 1
            D.append(new_row)
        data = np.array(D)
        data = np.transpose(data)  # The last line is Y value
    return data


# data partition
def data_partition(data, split_value, index):
    y = data[index]
    data_x = []
    for i in range(len(y)):
        if y[i] == split_value:
            data_x.append(data.transpose()[i])
    return np.array(data_x).transpose()


# check which attribute has the max mutual information, return that attribute, return -1 if max_mutual <= 0
def attr_iterator(label, data, attr_never):
    if data.size == 0: return -1
    max_mutual = 0  # Set max mutual information to 0
    attr_split = -1  # Preset the attribute to be split on to -1
    for i in attr_never:
        mutual_info = mutualInfo(label, data[i])
        if mutual_info > max_mutual:
            max_mutual = max(max_mutual, mutual_info)
            attr_split = i  # The attribute to be split on 0 or 1
    if max_mutual > 0:
        return attr_split
    else:
        return -1


# Function to form a decision tree - root: the tree root; tree_depth: depth of current tree;
# attr_never: attribute never used; data: The train data set; label_index: The y data index in data set
def DT_Former(root, tree_depth, attr_never, label_index, data):

    # Get the root attribute
    if root.val is None:
        attr_split = attr_iterator(data[label_index], data, attr_never)
        root.val = attr_split  # Assign the attribute to the root
        if attr_split != -1:
            attr_never.remove(attr_split)
            tree_depth = 1
        else:
            return

    # Cannot split when reaching the max depth or no attribute to use
    if tree_depth == depth or len(attr_never) == 0:
        root.left = Node()
        root.left.val = -1
        root.right = Node()
        root.right.val = -1
        return

    # split on 0, left children
    data_0 = data_partition(data, 0, root.val)  # Partition the data set
    attr_split = attr_iterator(data_0[label_index], data_0, attr_never)
    root.left = Node()
    root.left.val = attr_split  # Assign the attribute to left children node
    if attr_split != -1:  # Only split when mutual information > 0
        attr_never.remove(attr_split)
        tree_depth_new = tree_depth + 1  # Update the tree depth
        DT_Former(root.left, tree_depth_new, attr_never, label_index, data_0)
        attr_never.append(attr_split)  # add it back for using in 'split on 1'

    # split on 1, right children
    data_1 = data_partition(data, 1, root.val)  # Partition the data set
    attr_split = attr_iterator(data_1[label_index], data_1, attr_never)
    root.right = Node()
    root.right.val = attr_split  # Assign the attribute to right children node
    if attr_split != -1:  # Only split when mutual information > 0
        attr_never.remove(attr_split)
        tree_depth_new = tree_depth + 1  # Update the tree depth
        DT_Former(root.right, tree_depth_new, attr_never, label_index, data_1)
        attr_never.append(attr_split)  # add it back for later use

    return


# Put one train_data sample into the Decision Tree
def DT_Trainer(root, data):
    if data[len(data) - 1] == 0:  # Assign this train data result to the tree
        root.counter[0] += 1
    else:
        root.counter[1] += 1
    if root.val != -1:
        if data[root.val] == 0:
            DT_Trainer(root.left, data)
        else:
            DT_Trainer(root.right, data)
    return


# Predict the result for given data based on the Decision Tree
def DT_Predictor(root, data):
    if (root.val == -1):
        if root.counter[0] >= root.counter[1]:
            return 0
        else:
            return 1
    else:
        if data[root.val] == 0:
            return DT_Predictor(root.left, data)
        else:
            return DT_Predictor(root.right, data)


###################
### Main Method ###
###################

# Get the attribute list & name of attribute status referred as zero
with open(train_input) as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    attr_list = next(csv_data)
    num = len(attr_list)  # num = (num of attributes) + 1
    reference_as_zero = next(csv_data)
    # Get the two label names
    label_0 = reference_as_zero[num - 1]
    next_row = next(csv_data)
    while next_row[num - 1] == label_0:
        next_row = next(csv_data)
    label_1 = next_row[num - 1]

train_data = csv2matrix(train_input, reference_as_zero)  # The (0,1) matrix of train.csv data
test_data = csv2matrix(test_input, reference_as_zero)  # The (0,1) matrix of test.csv data

root = Node()
if depth == 0:  # Case where depth = 0, identify the root as leaf node
    root.val = -1
else:  # Form the decision tree
    DT_Former(root, 0, list(range(num - 1)), num - 1, train_data)

# Train the decision tree using train data
train_data = train_data.transpose()
for row_data in train_data:
    DT_Trainer(root, row_data)

# Predict for the train data and compute error rate
with open(train_out, 'w') as res:
    counter = 0
    error = 0
    for row_data in train_data:
        if DT_Predictor(root, row_data):
            print(label_1, file=res)
            if row_data[num - 1] != 1:
                error += 1
        else:
            print(label_0, file=res)
            if row_data[num - 1] != 0:
                error += 1

        counter += 1

error_train = error / counter

printer(root, 0)

# Predict for the test data and compute error rate
test_data = test_data.transpose()
counter = 0
error = 0
with open(test_out, 'w') as res:
    for row_data in test_data:
        if DT_Predictor(root, row_data):
            print(label_1, file=res)
            if row_data[num - 1] != 1:
                error += 1
        else:
            print(label_0, file=res)
            if row_data[num - 1] != 0:
                error += 1

        counter += 1

error_test = error / counter

# Print out the error rate report
with open(metric_out, 'w') as res:
    print("error(train): %.6f" % error_train, file=res)
    print("error(test): %.6f" % error_test, file=res)
