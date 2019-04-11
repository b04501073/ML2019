import numpy as np
import os
import random 
import sys
import csv
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import losses
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers.advanced_activations import PReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
import random
from keras.utils.np_utils import to_categorical

f_file = sys.argv[1]
datas = []
with open(f_file) as file:
    for line_id, line in enumerate(file):
        if line_id == 0:
            continue
        else:
            label, feature = line.split(',')
            feature = np.fromstring(feature, dtype=int, sep=' ')/225
            
            feature = feature.reshape((48, 48, 1))
            
            datas.append((feature, int(label)))
features, labels = zip(*datas)

features = np.asarray(features)
print(features.shape)
labels = to_categorical(np.asarray(labels, dtype = np.int32))

def gen_valid_set(features, labels, percent):
    length = len(features)
    val_length = int(length * percent)
    
    pairs = list(zip(features, labels))
    
    random.shuffle(pairs)
    
    features, labels = zip(*pairs)
    
    features = np.array(features)
    labels = np.array(labels)
    return features[:length - val_length], labels[:length - val_length], features[length - val_length:], labels[length - val_length:]

def model_build(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (5, 5), padding='same'))
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    return model

#validation set
tr_feats, tr_labels, val_feats, val_labels = gen_valid_set(features, labels, 0.1)

zoom_range = 0.2
train_gen = ImageDataGenerator(rotation_range=25, 
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=[1-zoom_range, 1+zoom_range],
                                horizontal_flip=True)
train_gen.fit(tr_feats)

input_shape = (48, 48, 1)
num_classes = 7
epochs = 400
batch_size = 128


model = model_build(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks = []
# modelcheckpoint = ModelCheckpoint("ckpt_center_norm/weights.{epoch:03d}-{val_acc:.5f}.h5", monitor='val_acc', save_best_only=True, mode='max', verbose=1)
# callbacks.append(modelcheckpoint)
# csv_logger = CSVLogger('cnn_log_center_norm.csv', separator=',', append=False)
# callbacks.append(csv_logger)
# es = EarlyStopping(monitor='val_loss', patience=70, verbose=1, mode='min')
# callbacks.append(es)
# model_feature = "3_Dense_CNN"

model.fit_generator(train_gen.flow(tr_feats, tr_labels, batch_size=batch_size),
                        steps_per_epoch=1*tr_feats.shape[0]//batch_size,
                        epochs=epochs,
#                         callbacks=callbacks,
                        validation_data=(val_feats, val_labels))

model.save("final_predict.h5")
score = model.evaluate(val_feats, val_labels)
print("\nValidation Acc for \"final_model\": {}\n".format(score[1]))