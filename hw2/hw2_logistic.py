import numpy as np
import csv
import sys

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std)
     
    return X, X_mean, X_std
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
    
def train_dev_split(X, y, dev_size=0.25):
    train_len = int(round(len(X)*(1-dev_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]
def sigmoid(z):
    f = 1. / (1. + np.exp(-z))
    bound = 1e-8
    return np.clip(f, bound, 1-bound)
def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc

def train(X_train, Y_train, regularize = True):
    # split a validation set
    dev_size = 0.1155
#     dev_size = 0.011
    X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_size = dev_size)
    
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_dev = Y_dev.reshape((len(Y_dev), 1))
    # Use 0 + 0*x1 + 0*x2 + ... for weight initialization
    w = np.zeros((X_train.shape[1],1))
#     b = np.zeros((1,1))

    if regularize:
        lamda = 0.001
    else:
        lamda = 0
    
    max_iter = 800  # max iteration number
    batch_size = 32 # number to feed in the model for average to avoid bias
    learning_rate = 0.00001  # how much the model learn for each step
    num_train = len(Y_train)
    num_dev = len(Y_dev)
    feature_num = len(X_train[0])
    step =1

    loss_train = []
    loss_validation = []
    train_acc = []
    dev_acc = []
    
    beta1 = 0.9
    beta2 = 0.999
    e = 10 ** -8

    Vdw = np.zeros(shape = (feature_num,1))

    Sdw = np.zeros(shape = (feature_num,1))
    
    
    for epoch in range(max_iter):
        # Random shuffle for each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)
        
        total_loss = 0.0
        # Logistic regression train with batch
        for idx in range(int(np.floor(len(Y_train)/batch_size))):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            # Find out the gradient of the loss
            z = np.dot(X, w)
            y_pred = sigmoid(z)
            grad = np.dot(X.T,(y_pred - Y))
#             grad[:len(grad)-2,] += lamda*w[:len(grad)-2,]
            
            Vdw = beta1 * Vdw + (1 - beta1) * grad
            Sdw = beta2 * Sdw + (1 - beta2) * grad ** 2
            
            Vdw_c = Vdw / (1 - beta1 ** (step))
            Sdw_c = Sdw / (1 - beta2 ** (step))
            
            w = w - learning_rate * Vdw_c / (np.sqrt(Sdw_c) + e)
            step += 1
            
        # Compute the loss and the accuracy of the training set and the validation set
        y_train_pred = sigmoid(np.dot(X_train, w))
        Y_train_pred = np.round(y_train_pred)
        
        train_acc.append(accuracy(Y_train_pred, Y_train))
        loss_train.append(-np.mean((Y_train * np.log(Y_train_pred + 10**(-10)) + (1 - Y_train) * np.log(1 - Y_train_pred + 10**(-10)))) + lamda * np.sum(np.square(w)))
        y_dev_pred = sigmoid(np.dot(X_dev, w))
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(accuracy(Y_dev_pred, Y_dev))
        loss_validation.append(-np.mean((Y_dev * np.log(Y_dev_pred + 10**(-10)) + (1 - Y_dev) * np.log(1 - Y_dev_pred + 10**(-10)))) + lamda * np.sum(np.square(w)))
    
    return w, loss_train, loss_validation, train_acc, dev_acc  # return loss for plotting
#main

X_train_fpath = sys.argv[1]
Y_train_fpath = sys.argv[2]
X_train = np.genfromtxt(X_train_fpath, delimiter=',', skip_header=1)
Y_train = np.genfromtxt(Y_train_fpath, delimiter=',', skip_header=1)
feat_1 = (np.power((X_train[:, 0], X_train[:, 1], X_train[:, 3], X_train[:, 4], X_train[:, 5]) , 2.0)).T
feat_2 = (np.power((X_train[:, 0], X_train[:, 1], X_train[:, 3], X_train[:, 4], X_train[:, 5]) , 3.0)).T
feat_3 = (np.power((X_train[:, 0], X_train[:, 1], X_train[:, 3], X_train[:, 4], X_train[:, 5]) , 4.0)).T
feat_4 = (np.power((X_train[:, 0], X_train[:, 1], X_train[:, 3], X_train[:, 4], X_train[:, 5]) , 5.0)).T
X_train = np.concatenate((X_train, feat_1, feat_2, feat_3, feat_4), axis = 1)

col = [0,1,3,4,5,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125]
X_train, X_mean, X_std = _normalize_column_normal(X_train, specified_column=col)
X_train = np.concatenate((X_train, np.ones((len(X_train),1))), axis = 1)

# w, loss_train, loss_validation, train_acc, dev_acc = train(X_train, Y_train)
w = np.load("model/model_logistic.npy")

X_test_fpath = sys.argv[3]
X_test = np.genfromtxt(X_test_fpath, delimiter=',', skip_header=1)
feat_test_1 = (np.power((X_test[:, 0], X_test[:, 1], X_test[:, 3], X_test[:, 4], X_test[:, 5]) , 2.0)).T
feat_test_2 = (np.power((X_test[:, 0], X_test[:, 1], X_test[:, 3], X_test[:, 4], X_test[:, 5]) , 3.0)).T
feat_test_3 = (np.power((X_test[:, 0], X_test[:, 1], X_test[:, 3], X_test[:, 4], X_test[:, 5]) , 4.0)).T
feat_test_4 = (np.power((X_test[:, 0], X_test[:, 1], X_test[:, 3], X_test[:, 4], X_test[:, 5]) , 5.0)).T
X_test = np.concatenate((X_test, feat_test_1, feat_test_2, feat_test_3, feat_test_4), axis = 1)
X_test, _, _= _normalize_column_normal(X_test, train=False, specified_column = col, X_mean=X_mean, X_std=X_std)
X_test = np.concatenate((X_test, np.ones((len(X_test), 1))), axis = 1)

z = np.dot(X_test, w)
y_test_pred = np.round(sigmoid(z))
y_test_pred = y_test_pred.reshape((len(y_test_pred)))

output_path = sys.argv[4]
output = open(output_path, 'w', newline='')
writer = csv.writer(output)
writer.writerow(['id', 'label'])
for i in range(len(y_test_pred)):
    if (y_test_pred[i] == 0):
        writer.writerow([str(i+1), '0'])
    else:
        writer.writerow([str(i+1), '1'])
output.close()