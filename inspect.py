import sys
import numpy as np
import csv
import math

with open(sys.argv[1]) as csvfile:
    csv_data = csv.reader(csvfile, delimiter=',')
    num = len(next(csv_data))  # num = (num of attributes) + 1
    name_as_zero = next(csv_data)
    D = [[0] * num]
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

# Calculate the label entropy and the error rate
y_data = data[num - 1]
percent = sum(y_data) / len(y_data)
if percent == 0 or percent == 1:
    entropy = 0
else:
    entropy = -percent * (math.log2(percent)) - (1 - percent) * (math.log2(1 - percent))

error_rate = percent if percent < 0.5 else 1 - percent

# Write to output file"
with open(sys.argv[2], 'w') as res:
    print('entropy: %.12f' % entropy, file=res)
    print('error: %.12f' % error_rate, file=res)
