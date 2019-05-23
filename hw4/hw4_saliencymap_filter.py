from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
import keras
import sys
from keras.models import Model
from keras.utils import get_file
import os
# K.set_learning_phase(1)

f_file = sys.argv[1]
path = sys.argv[2]
#f_file = "train.csv"
#path = "figs"
datas = []
with open(f_file) as file:
    for line_id, line in enumerate(file):
        if line_id == 0:
            continue
        else:
            label, feature = line.split(',')
            feature = np.fromstring(feature, dtype=int, sep=' ')/255
            
            feature = feature.reshape((48, 48, 1))
            
            datas.append((feature, int(label)))
x_train, x_label = zip(*datas)
x_train = np.array(x_train)

model_web = "https://www.dropbox.com/s/rfwi4zsfnm94xsn/final_predict.h5?dl=1"
current_path = os.path.abspath('.')
get_file(current_path+"/final_model.h5", origin=model_web)
model_name = "final_model.h5"
model = load_model(model_name)
layer_dict = dict([(layer.name, layer) for layer in model.layers])

class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients
    
# fig, ax = plt.subplots(7, 3, figsize = (16, 16))  
imgs_idx = [0,299,2,7,3,15,4]
for i in range(7):
    img = x_train[imgs_idx[i]]
    vanilla = GradientSaliency(model, x_label[i])
    saliency = vanilla.get_mask(img)
    saliency = np.absolute(saliency)
    plt.imsave(path + '/fig1_'+ str(i)+".jpg", saliency.reshape((48,48)), cmap='jet')
    
K.set_learning_phase(1)
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def vis_img_in_filter(img = np.array(x_train[1000]).reshape((1, 48, 48, 1)).astype(np.float64), 
                      layer_name = 'conv2d_26'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    if layer_output.shape[3] >= 63:
        plot_x, plot_y = 8, 8
    elif layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    fig.suptitle('Input image of %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
    plt.savefig(path + "/fig2_1.jpg")
    
vis_img_in_filter()

layer_output = layer_dict["conv2d_26"].output
mid_model = Model(inputs=model.input, outputs=layer_output)
img = np.array(x_train[1000]).reshape((1, 48, 48, 1)).astype(np.float64)
mid_output = mid_model.predict(img)[0]
plot_x, plot_y = 8, 8
fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
fig.suptitle('Input image of conv2d_26\'s filters')
fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])

img_ascs = []
for i in range(mid_output.shape[2]):
    img_ascs.append(mid_output[:,:,i])
for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
    ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
    ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
plt.savefig(path + "/fig2_2.jpg")
