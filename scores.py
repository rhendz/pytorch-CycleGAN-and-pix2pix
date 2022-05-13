# Compute the Perfomance metrics for the GAN architectures

# Citation https://wandb.ai/ayush-thakur/gan-evaluation/reports/How-to-Evaluate-GANs-using-Frechet-Inception-Distance-FID---Vmlldzo0MTAxOTI


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
import numpy as np 
import tensorflow as tf
from tqdm import tqdm
import math
import pickle

# The pretrained model on imagenet
inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg')


# Calculate the embeddings of each image, dataloader is all your images, count is the number of images. 
def compute_embeddings(dataloader, count):
    image_embeddings = []

    for _ in tqdm(range(count)):
        images = next(iter(dataloader))
        embeddings = inception_model.predict(images)

        image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = np.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
       covmean = covmean.real
     # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def fid(real_images, generated_images):
    count = len(real_images)

    # compute embeddings for real images
    real_images_embeddings = compute_embeddings(real_images, count)

    # compute embeddings for generated images
    generated_images_embeddings = compute_embeddings(generated_images, count)

    # calulate the fid score between the embeddings
    fid = calculate_fid(real_images_embeddings, generated_images_embeddings)

    return fid

def test():
    print("Testing Performance")

test()