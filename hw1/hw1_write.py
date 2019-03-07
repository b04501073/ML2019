import numpy as np
import sys
import csv
w = np.load('weight_adam.npy')
mean  = np.load('mean.npy')
std  = np.load('std.npy')

test_x = []
test_y = []

text = open(sys.argv[1], 'r', encoding = 'big5')

n_row2 = 0
n_data = 0


row = csv.reader(text, delimiter = ',')
for r in row:
    r_row = n_row2 % 18
    if r_row == 0:
        test_x.append([])
        n_data += 1
        sum_row = 0
    
    if r_row != 0 and r_row != 10 and r_row != 16 and r_row != 17 and r_row != 4:
        for i in range(2,11):
            if r[i] != "NR":
                test_x[n_data-1].append(float(r[i]))
            else:
                test_x[n_data-1].append(float(0))
        sum_row += 1
    n_row2 += 1
text.close()
test_x = np.array(test_x)
test_y = np.array(test_y)

for i in range(test_x.shape[0]):        ##Normalization
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

test_x = np.concatenate((np.ones([len(test_x),1]), test_x), axis = 1)
answer = test_x.dot(w)
output = open(sys.argv[2], 'w', newline='')
writer = csv.writer(output)
writer.writerow(['id', 'value'])
for i in range(240):
    pred = float(answer[i])
    if pred < 0:
        pred = 0
    writer.writerow(['id_' + str(i), pred])
output.close()
