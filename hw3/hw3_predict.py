from keras.models import load_model
from keras.utils import get_file
import csv
import numpy as np
import sys
import tensorflow as tf
import os

model_web = "https://www.dropbox.com/s/rfwi4zsfnm94xsn/final_predict.h5?dl=1"
current_path = os.path.abspath('.')
get_file(current_path+"/final_model.h5", origin=model_web)

model = load_model("final_model.h5")

def read_dataset(data_path = ""):
    # num_data = 0
    datas = []
    read_path = "data/test.csv"

    with open(read_path) as file:
        for line_id,line in enumerate(file):
            if(line_id == 0):
                continue
            _,feat = line.split(',')
            feat = np.fromstring(feat, dtype=int, sep=' ') / 255.
            # print(feat)
            feat = np.reshape(feat, (48, 48, 1))

            datas.append(feat)

    feats = datas
    feats = np.asarray(feats)
    return feats
    
te_feats = read_dataset(data_path=sys.argv[1])
y_prob = model.predict(te_feats)
prd_class = y_prob.argmax(axis=-1)

pred_file = sys.argv[2]
with open(pred_file, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(te_feats)):
        csv_writer.writerow([i]+[prd_class[i]])