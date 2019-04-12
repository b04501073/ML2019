import numpy as np
from lime import lime_image
from skimage.segmentation import slic,mark_boundaries
import matplotlib.pyplot as plt
from keras.models import load_model
import skimage
import sys

f_file = sys.argv[1]
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
x_train = np.asarray(x_train)

model_web = "https://www.dropbox.com/s/rfwi4zsfnm94xsn/final_predict.h5?dl=1"
current_path = os.path.abspath('.')
get_file(current_path+"/final_model.h5", origin=model_web)
model_name = "final_predict.h5"
model = load_model(model_name)

fig, ax = plt.subplots(7, 1, figsize = (16, 16))
imgs_idx = [0,299,2,7,3,15,4]
x_train_rgb = []
x_train_rgb_label = []
for i in range(7):
    
    img = x_train[imgs_idx[i]]
    newimg = skimage.color.gray2rgb(img)
    newimg = newimg.astype(float)
    
    newimg = newimg.reshape((48,48,3))
    ax[i].imshow(newimg)
    
    
    x_train_rgb.append(newimg)
    x_train_rgb_label.append(x_label[imgs_idx[i]])
    
def predict(img_input):
    img = img_input[0]
    img = img[:,:,0]
    print(img.shape)
    y_prob = model.predict(img.reshape(1,48,48,1))
    return y_prob
def segmentation(img_input):
#     img = img_input[:,0]
#     img = img.reshape((48,48))
    segments = slic(img_input, n_segments=40, compactness=1)
#     out2=mark_boundaries(img,segments)
    return segments

# Initiate explainer instance
idx = 0
explainer = lime_image.LimeImageExplainer()
path = sys.argv[2]
for i in range(7):
    explaination = explainer.explain_instance(
                                image=x_train_rgb[0], 
                                classifier_fn=predict,
                                segmentation_fn=segmentation
                            )

    # # Get processed image
    image, mask = explaination.get_image_and_mask(
                                    label=x_train_rgb_label[idx],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=7,
                                    min_weight=0.0
                                )
    plt.imsave(path+'fig3_'+str(i)+".jpg", image)
