from skimage import io
import numpy as np
import sys

imges_path = sys.argv[1]
input_image = sys.argv[2]
output_image_path = sys.argv[3]

images = []
for i in range(415):
    img = io.imread(imges_path + str(i)+".jpg")
    images.append(img.flatten())
images = np.array(images).astype('float32')

mean = np.mean(images, axis = 0)  
images -= mean

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
def pca2(x_tk,n_pc):
    u,d,vh = np.linalg.svd(x_tk,full_matrices = False)
    pcs_nk = vh[:n_pc]
    return pcs_nk, np.dot(x_tk,pcs_nk.T), d

e_vec, final, e_val = pca2(images,5)

img = io.imread(imges_path+input_image)
img = (np.array(img)).reshape((-1)) - mean
r_img = np.dot(np.dot(img, e_vec.T), e_vec)

img1 = (process(r_img + mean)).reshape((600,600,3)).astype(np.uint8)
io.imsave(output_image_path,img1)