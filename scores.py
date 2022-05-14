# Compute the Perfomance metrics for the GAN architectures

# Source 1: https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI
# Source 2: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

# Trying to evaluate how well you GAN is doing can be a bit difficult since the loss does not coorelate with how well the model is performing.
# The most basic way is to see the output of the image after a certain amount of epochs and visually inspect the image. 
# There are two metrics which are commonly used fidelity and diversity.
# We want to be able to generate different types of high quality images.
# Trying to calculate these metrics can be difficult as well since how do we know if images are high in quality.
# One reliable metric used to determine this is Feature Distance.
# With feature distance, a pre-trained image classification model and the activation of an intermediate layer are used to compute the metric.

# FID (Frechet Inception Distance)
# Frechet Distance is a measure of similarity between curves that takes into account the location and ordering of the points along the curves.

#
from matplotlib import image
import numpy as np 
import tensorflow as tf
from torch import QInt32Storage
from tqdm import tqdm
import math
import pickle
from PIL import Image
from numpy import asarray
import os

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize

def load_test_images(directory):
    images = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        real_fn = f.replace('\\', '/')
        image = Image.open(real_fn)
        data = asarray(image)
        images.append(data)
    
    return np.array(images)

# The pretrained model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))


# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def fid(real_images, generated_images):
    real_images = real_images.astype('float32')
    generated_images = generated_images.astype('float32')
    # resize images
    real_images = scale_images(real_images, (299,299,3))
    generated_images = scale_images(generated_images, (299,299,3))
    # print('Scaled', real_images.shape, generated_images.shape)
    # pre-process images
    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)
    # fid between real and generated
    fid = calculate_fid(model, real_images, real_images)
    print('FID (same): %.3f' % fid)
    # fid between real and generated
    fid = calculate_fid(model, real_images, generated_images)
    print('FID: %.3f' % fid)

# SSIM (Structural SIMilarity)
from skimage.metrics import structural_similarity

def ssim(real_images, generated_images):
    # fid between images1 and images1
    (score_same, diff_same) = structural_similarity(real_images, real_images, full=True, channel_axis=3)
    print('ssim (same): %.3f' % score_same)
    # fid between images1 and images2
    (score, diff) = structural_similarity(real_images, generated_images, full=True, channel_axis=3)
    print('ssim: %.3f' % score)


def test():
    print("Testing FID Score")
    real_images = load_test_images("./datasets/apple2orange/testA")
    generated_images = load_test_images("./datasets/apple2orange/trainA")
    generated_images = generated_images[:len(real_images)]
    fid(real_images.copy(), generated_images.copy())
    print("Testing SSIM Score")
    ssim(real_images.copy(), generated_images.copy())

test()