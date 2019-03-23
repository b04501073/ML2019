import numpy as np
import csv
import sys
from numpy.linalg import inv

class datamanager():
    def __init__(self):
        self.data = {}
        
    def read(self, name, path):
        with open(path,newline = '') as csvfile:
            rows = np.array(list(csv.reader(csvfile))[1:] ,dtype = float)  
            if name == 'X_train':
                feat_1 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 2.0)).T
                feat_2 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 3.0)).T
                feat_3 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 4.0)).T
                feat_4 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 5.0)).T
                rows = np.concatenate((rows, feat_1, feat_2, feat_3, feat_4), axis = 1)
                self.mean = np.mean(rows,axis = 0).reshape(1,-1)
                self.std = np.std(rows,axis = 0).reshape(1,-1)
                self.theta = np.ones((rows.shape[1] + 1,1),dtype = float)
                
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std  

            elif name == 'X_test': 
                feat_1 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 2.0)).T
                feat_2 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 3.0)).T
                feat_3 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 4.0)).T
                feat_4 = (np.power((rows[:, 0], rows[:, 1], rows[:, 3], rows[:, 4], rows[:, 5]) , 5.0)).T
                rows = np.concatenate((rows, feat_1, feat_2, feat_3, feat_4), axis = 1)
                for i in range(rows.shape[0]):
                    rows[i,:] = (rows[i,:] - self.mean) / self.std 

            self.data[name] = rows 
        
    def find_theta(self):
        class_0_id = []
        class_1_id = []
        for i in range(self.data['Y_train'].shape[0]):
            if self.data['Y_train'][i][0] == 0:
                class_0_id.append(i)
            else:
                class_1_id.append(i)

        class_0 = self.data['X_train'][class_0_id]
        class_1 = self.data['X_train'][class_1_id] 

        mean_0 = np.mean(class_0,axis = 0)
        mean_1 = np.mean(class_1,axis = 0)  

        n = class_0.shape[1]
        cov_0 = np.zeros((n,n))
        cov_1 = np.zeros((n,n))
        
        for i in range(class_0.shape[0]):
            cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]

        for i in range(class_1.shape[0]):
            cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [(class_1[i] - mean_1)]) / class_1.shape[0]

        cov = (cov_0*class_0.shape[0] + cov_1*class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])
 
        self.w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)) )
        self.b =  (- 0.5)* (mean_0).dot(inv(cov)).dot(mean_0) + 0.5 * (mean_1).dot(inv(cov)).dot(mean_1) + np.log(float(class_0.shape[0]) / class_1.shape[0]) 

        result = self.func(self.data['X_train'])
        answer = self.predict(result)
        
    def func(self,x):
        arr = np.empty([x.shape[0],1],dtype=float)
        for i in range(x.shape[0]):
            z = x[i,:].dot(self.w) + self.b
            z *= (-1)
            arr[i][0] = 1 / (1 + np.exp(z))
        return np.clip(arr, 1e-8, 1-(1e-8))
    
    def predict(self,x):
        ans = np.ones([x.shape[0],1],dtype=int)
        for i in range(x.shape[0]):
            if x[i] > 0.5:
                ans[i] = 0; 
        return ans

    def write_file(self,path):
        result = self.func(self.data['X_test'])
        answer = self.predict(result)
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile) 
            writer.writerow(['id','label']) 
            for i in range(answer.shape[0]):
                writer.writerow([i+1,answer[i][0]])
                
dm = datamanager()
dm.read("X_train", sys.argv[1])
dm.read("Y_train", sys.argv[2])
dm.read("X_test", sys.argv[3])
dm.find_theta()
dm.write_file(sys.argv[4])