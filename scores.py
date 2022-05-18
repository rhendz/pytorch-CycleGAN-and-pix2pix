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
from msilib.schema import Directory
from matplotlib import image
from matplotlib.style import available
import numpy as np 
import tensorflow as tf
from torch import QInt32Storage
from tqdm import tqdm
import math
import pickle
from PIL import Image
from numpy import asarray, average
import os
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

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
    # fid = calculate_fid(model, real_images, real_images)
    # print('FID (same): %.3f' % fid)
    # fid between real and generated
    fid = calculate_fid(model, real_images, generated_images)
    # print('FID: %.3f' % fid)
    return fid

# SSIM (Structural SIMilarity)
from skimage.metrics import structural_similarity

def ssim(real_images, generated_images):
    # fid between images1 and images1
    # (score_same, diff_same) = structural_similarity(real_images, real_images, full=True, channel_axis=3)
    # print('ssim (same): %.3f' % score_same)
    # fid between images1 and images2
    (score, diff) = structural_similarity(real_images, generated_images, full=True, channel_axis=2)
    # print('ssim: %.3f' % score)
    return score

# Plots the scores for two sets of images generated from two CycleGANs with the same number of epochs. 
def individual_scores():
    epochs = "25"
    dir = "results/apple2orange_resized_32_no_vae/test_latest/images_"+epochs+"/"
    dir2 = "results/apple2orange_resized_32_vae/test_latest/images_"+epochs+"/"
    no_vae_fake_A = []
    no_vae_fake_B = []
    no_vae_real_A = []
    no_vae_real_B = []
    no_vae_rec_A = []
    no_vae_rec_B = []
    vae_fake_A = []
    vae_fake_B = []
    vae_real_A = []
    vae_real_B = []
    vae_rec_A = []
    vae_rec_B = []

    for file in os.listdir(dir):
        #print(file)
        if("fake_A" in file):
            no_vae_fake_B.append(file)
        if("fake_B" in file):
            no_vae_fake_A.append(file)
        if("real_A" in file):
            no_vae_real_A.append(file)
        if("real_B" in file):
            no_vae_real_B.append(file)
        if("rec_A" in file):
            no_vae_rec_A.append(file)
        if("rec_B" in file):
            no_vae_rec_B.append(file)

    for file in os.listdir(dir2):
        #print(file)
        if("fake_A" in file):
            vae_fake_B.append(file)
        if("fake_B" in file):
            vae_fake_A.append(file)
        if("real_A" in file):
            vae_real_A.append(file)
        if("real_B" in file):
            vae_real_B.append(file)
        if("rec_A" in file):
            vae_rec_A.append(file)
        if("rec_B" in file):
            vae_rec_B.append(file)

    no_vae_fake_A.sort()
    no_vae_fake_B.sort()
    no_vae_real_A.sort()
    no_vae_real_B.sort()
    no_vae_rec_A.sort()
    no_vae_rec_B.sort()

    vae_fake_A.sort()
    vae_fake_B.sort()
    vae_real_A.sort()
    vae_real_B.sort()
    vae_rec_A.sort()
    vae_rec_B.sort()

    no_vae_fake_A_images = []
    no_vae_fake_B_images = []
    no_vae_real_A_images = []
    no_vae_real_B_images = []
    no_vae_rec_A_images = []
    no_vae_rec_B_images = []
    vae_fake_A_images = []
    vae_fake_B_images = []
    vae_real_A_images = []
    vae_real_B_images = []
    vae_rec_A_images = []
    vae_rec_B_images = []


    # Calculate SSIM for the Apples dataset 

    no_vae_ssim_real_fake_A = []
    no_vae_ssim_rec_fake_A = []
    no_vae_ssim_real_real_A = []
    no_vae_ssim_real_rec_A = []

    vae_ssim_real_fake_A = []
    vae_ssim_rec_fake_A = []
    vae_ssim_real_real_A = []
    vae_ssim_real_rec_A = []

    # Calculates all the SSIM scores

    for i in range(len(no_vae_real_A)):

        # Open all the images and save them as numpy arrays 

        no_vae_real_A_image = Image.open(os.path.join(dir, no_vae_real_A[i]))
        no_vae_real_A_image = asarray(no_vae_real_A_image)
        no_vae_fake_A_image = Image.open(os.path.join(dir, no_vae_fake_A[i]))
        no_vae_fake_A_image = asarray(no_vae_fake_A_image)
        no_vae_rec_A_image = Image.open(os.path.join(dir, no_vae_rec_A[i]))
        no_vae_rec_A_image = asarray(no_vae_rec_A_image)

        vae_real_A_image = Image.open(os.path.join(dir2, no_vae_real_A[i]))
        vae_real_A_image = asarray(vae_real_A_image)
        vae_fake_A_image = Image.open(os.path.join(dir2, no_vae_fake_A[i]))
        vae_fake_A_image = asarray(vae_fake_A_image)
        vae_rec_A_image = Image.open(os.path.join(dir2, no_vae_rec_A[i]))
        vae_rec_A_image = asarray(vae_rec_A_image)

        no_vae_fake_A_images.append(no_vae_real_A_image)
        no_vae_real_A_images.append(no_vae_fake_A_image)
        no_vae_rec_A_images.append(no_vae_rec_A_image)
        vae_fake_A_images.append(vae_real_A_image)
        vae_real_A_images.append(vae_fake_A_image)
        vae_rec_A_images.append(vae_rec_A_image)

        # Calculate the SSIM scores between the images

        no_vae_ssim_real_fake_A.append(ssim(no_vae_real_A_image.copy(), no_vae_fake_A_image.copy()))
        no_vae_ssim_rec_fake_A.append(ssim(no_vae_rec_A_image.copy(), no_vae_fake_A_image.copy()))
        no_vae_ssim_real_real_A.append(ssim(no_vae_real_A_image.copy(), no_vae_real_A_image.copy()))
        no_vae_ssim_real_rec_A.append(ssim(no_vae_real_A_image.copy(), no_vae_rec_A_image.copy()))

        vae_ssim_real_fake_A.append(ssim(vae_real_A_image.copy(), vae_fake_A_image.copy()))
        vae_ssim_rec_fake_A.append(ssim(vae_rec_A_image.copy(), vae_fake_A_image.copy()))
        vae_ssim_real_real_A.append(ssim(vae_real_A_image.copy(), vae_real_A_image.copy()))
        vae_ssim_real_rec_A.append(ssim(vae_real_A_image.copy(), vae_rec_A_image.copy()))

    # Plot the scores as a line plot and box and whisker plot 
    figs = (12, 8)
    xs = range(len(no_vae_ssim_real_fake_A))
    figure(figsize=figs, dpi=100)
    plt.plot(xs, no_vae_ssim_real_real_A, label="No_VAE_Real_Real_A")
    plt.plot(xs, no_vae_ssim_real_fake_A, label="No_VAE_Real_Fake_A")
    plt.plot(xs, no_vae_ssim_rec_fake_A, label="No_VAE_Rec_Fake_A")
    plt.plot(xs, vae_ssim_real_real_A, label="VAE_Real_Real_A")
    plt.plot(xs, vae_ssim_real_fake_A, label="VAE_Real_Fake_A")
    plt.plot(xs, vae_ssim_rec_fake_A, label="VAE_Rec_Fake_A")
    # plt.plot(xs, fids_real_rec_A)
    plt.legend()
    plt.savefig("SSIM_A_Plot_"+epochs+".png")
    plt.show()
    
    plt.close()
    figure(figsize=figs, dpi=100)
    plt.boxplot([no_vae_ssim_real_real_A, no_vae_ssim_real_fake_A, no_vae_ssim_rec_fake_A,
                vae_ssim_real_real_A, vae_ssim_real_fake_A, vae_ssim_rec_fake_A]
                , labels = ["No_VAE_Real_Real_A", "No_VAE_Real_Fake_A", "No_VAE_Rec_Fake_A", 
                "VAE_Real_Real_A", "VAE_Real_Fake_A", "VAE_Rec_Fake_A"], showmeans=True)
    plt.savefig("SSIM_A_Boxplot_"+epochs+".png")
    plt.show()
    
    figure(figsize=figs, dpi=100)
    print("No_VAE_Real_Fake_A: ", average(no_vae_ssim_real_fake_A))
    print("No_VAE_Rec_Fake_A: ", average(no_vae_ssim_rec_fake_A))
    print("VAE_Real_Fake_A: ", average(vae_ssim_real_fake_A))
    print("VAE_Rec_Fake_A: ", average(vae_ssim_rec_fake_A))
    print("FID No_VAE_Real_Fake_A: ", fid(np.array(no_vae_real_A_images), np.array(no_vae_fake_A_images)))
    print("FID No_VAE_Rec_Fake_A: ", fid(np.array(no_vae_rec_A_images), np.array(no_vae_fake_A_images)))
    print("FID VAE_Real_Fake_A: ", fid(np.array(vae_real_A_images), np.array(vae_fake_A_images)))
    print("FID VAE_Rec_Fake_A: ", fid(np.array(vae_rec_A_images), np.array(vae_fake_A_images)))

    # plt.close()

    # Calculate SSIM for the Oranges Dataset 

    no_vae_ssim_real_fake_B = []
    no_vae_ssim_rec_fake_B = []
    no_vae_ssim_real_real_B = []
    no_vae_ssim_real_rec_B = []

    vae_ssim_real_fake_B = []
    vae_ssim_rec_fake_B = []
    vae_ssim_real_real_B = []
    vae_ssim_real_rec_B = []

    for i in range(len(no_vae_real_B)):

        # Open all the images and save them as numpy arrays 

        no_vae_real_B_image = Image.open(os.path.join(dir, no_vae_real_B[i]))
        no_vae_real_B_image = asarray(no_vae_real_B_image)
        no_vae_fake_B_image = Image.open(os.path.join(dir, no_vae_fake_B[i]))
        no_vae_fake_B_image = asarray(no_vae_fake_B_image)
        no_vae_rec_B_image = Image.open(os.path.join(dir, no_vae_rec_B[i]))
        no_vae_rec_B_image = asarray(no_vae_rec_B_image)

        vae_real_B_image = Image.open(os.path.join(dir2, no_vae_real_B[i]))
        vae_real_B_image = asarray(vae_real_B_image)
        vae_fake_B_image = Image.open(os.path.join(dir2, no_vae_fake_B[i]))
        vae_fake_B_image = asarray(vae_fake_B_image)
        vae_rec_B_image = Image.open(os.path.join(dir2, no_vae_rec_B[i]))
        vae_rec_B_image = asarray(vae_rec_B_image)

        no_vae_fake_B_images.append(no_vae_real_B_image)
        no_vae_real_B_images.append(no_vae_fake_B_image)
        no_vae_rec_B_images.append(no_vae_rec_B_image)
        vae_fake_B_images.append(vae_real_B_image)
        vae_real_B_images.append(vae_fake_B_image)
        vae_rec_B_images.append(vae_rec_B_image)

        # Calculate the SSIM scores between the images

        no_vae_ssim_real_fake_B.append(ssim(no_vae_real_B_image.copy(), no_vae_fake_B_image.copy()))
        no_vae_ssim_rec_fake_B.append(ssim(no_vae_rec_B_image.copy(), no_vae_fake_B_image.copy()))
        no_vae_ssim_real_real_B.append(ssim(no_vae_real_B_image.copy(), no_vae_real_B_image.copy()))
        no_vae_ssim_real_rec_B.append(ssim(no_vae_real_B_image.copy(), no_vae_rec_B_image.copy()))

        vae_ssim_real_fake_B.append(ssim(vae_real_B_image.copy(), vae_fake_B_image.copy()))
        vae_ssim_rec_fake_B.append(ssim(vae_rec_B_image.copy(), vae_fake_B_image.copy()))
        vae_ssim_real_real_B.append(ssim(vae_real_B_image.copy(), vae_real_B_image.copy()))
        vae_ssim_real_rec_B.append(ssim(vae_real_B_image.copy(), vae_rec_B_image.copy()))

    # Plot the scores as a line plot and box and whisker plot 

    xs = range(len(no_vae_ssim_real_fake_B))
    plt.plot(xs, no_vae_ssim_real_real_B, label="No_VAE_Real_Real_B")
    plt.plot(xs, no_vae_ssim_real_fake_B, label="No_VAE_Real_Fake_B")
    plt.plot(xs, no_vae_ssim_rec_fake_B, label="No_VAE_Rec_Fake_B")
    plt.plot(xs, vae_ssim_real_real_B, label="VAE_Real_Real_B")
    plt.plot(xs, vae_ssim_real_fake_B, label="VAE_Real_Fake_B")
    plt.plot(xs, vae_ssim_rec_fake_B, label="VAE_Rec_Fake_B")
    # plt.plot(xs, fids_real_rec_A)
    plt.legend()
    plt.savefig("SSIM_B_Plot_"+epochs+".png")
    plt.show()
    
    # plt.close()
    figure(figsize=figs, dpi=100)
    plt.boxplot([no_vae_ssim_real_real_B, no_vae_ssim_real_fake_B, no_vae_ssim_rec_fake_B,
                vae_ssim_real_real_B, vae_ssim_real_fake_B, vae_ssim_rec_fake_B]
                , labels = ["No_VAE_Real_Real_B", "No_VAE_Real_Fake_B", "No_VAE_Rec_Fake_B", 
                "VAE_Real_Real_B", "VAE_Real_Fake_B", "VAE_Rec_Fake_B"], showmeans=True)

    print("No_VAE_Real_Fake_B: ", average(no_vae_ssim_real_fake_B))
    print("No_VAE_Rec_Fake_B: ", average(no_vae_ssim_rec_fake_B))
    print("VAE_Real_Fake_B: ", average(vae_ssim_real_fake_B))
    print("VAE_Rec_Fake_B: ", average(vae_ssim_rec_fake_B))
    print("FID No_VAE_Real_Fake_B: ", fid(np.array(no_vae_real_B_images), np.array(no_vae_fake_B_images)))
    print("FID No_VAE_Rec_Fake_B: ", fid(np.array(no_vae_rec_B_images), np.array(no_vae_fake_B_images)))
    print("FID VAE_Real_Fake_B: ", fid(np.array(vae_real_B_images), np.array(vae_fake_B_images)))
    print("FID VAE_Rec_Fake_B: ", fid(np.array(vae_rec_B_images), np.array(vae_fake_B_images)))
    plt.savefig("SSIM_B_Boxplot_"+epochs+".png")
    plt.show()
    
# Plots the scores for real and fake images for multiple CycleGANs trained every fifth epoch
def per_epoch():
    box_plot_vals = []
    box_plot_names = []
    for ep in range(5, 80, 5):
        eps = str(ep)
        dir = "results/apple2orange_resized_32_no_vae_checkpoint_"+eps+"/results/apple2orange_resized_32_no_vae/test_latest/images/"
        dir2 = "results/apple2orange_resized_32_vae_checkpoint_"+eps+"/results/apple2orange_resized_32_vae/test_latest/images/"
        no_vae_fake_A = []
        no_vae_real_A = []
        vae_fake_A = []
        vae_real_A = []

        for file in os.listdir(dir):
            if("fake_B" in file):
                no_vae_fake_A.append(file)
            if("real_A" in file):
                no_vae_real_A.append(file)
        for file in os.listdir(dir2):
            if("fake_B" in file):
                vae_fake_A.append(file)
            if("real_A" in file):
                vae_real_A.append(file)

        no_vae_fake_A.sort()
        no_vae_real_A.sort()

        vae_fake_A.sort()
        vae_real_A.sort()

        no_vae_fake_A_images = []
        no_vae_real_A_images = []
        vae_fake_A_images = []
        vae_real_A_images = []


        # Calculate SSIM for the Apples dataset 

        no_vae_ssim_real_fake_A = []

        vae_ssim_real_fake_A = []

        # Calculates all the SSIM scores

        for i in range(len(no_vae_real_A)):

            # Open all the images and save them as numpy arrays 

            no_vae_real_A_image = Image.open(os.path.join(dir, no_vae_real_A[i]))
            no_vae_real_A_image = asarray(no_vae_real_A_image)
            no_vae_fake_A_image = Image.open(os.path.join(dir, no_vae_fake_A[i]))
            no_vae_fake_A_image = asarray(no_vae_fake_A_image)

            vae_real_A_image = Image.open(os.path.join(dir2, no_vae_real_A[i]))
            vae_real_A_image = asarray(vae_real_A_image)
            vae_fake_A_image = Image.open(os.path.join(dir2, no_vae_fake_A[i]))
            vae_fake_A_image = asarray(vae_fake_A_image)

            no_vae_fake_A_images.append(no_vae_real_A_image)
            no_vae_real_A_images.append(no_vae_fake_A_image)
            vae_fake_A_images.append(vae_real_A_image)
            vae_real_A_images.append(vae_fake_A_image)

            # Calculate the SSIM scores between the images

            no_vae_ssim_real_fake_A.append(ssim(no_vae_real_A_image.copy(), no_vae_fake_A_image.copy()))

            vae_ssim_real_fake_A.append(ssim(vae_real_A_image.copy(), vae_fake_A_image.copy()))
        
        box_plot_vals.append(no_vae_ssim_real_fake_A)
        box_plot_vals.append(vae_ssim_real_fake_A)
        box_plot_names.append("No_VAE_"+eps)
        box_plot_names.append("VAE_"+eps)
        print("No_VAE_Real_Fake_A_"+eps+": ", average(no_vae_ssim_real_fake_A))
        print("VAE_Real_Fake_A_"+eps+": ", average(vae_ssim_real_fake_A))
        print("FID No_VAE_Real_Fake_A_"+eps+": ", fid(np.array(no_vae_real_A_images), np.array(no_vae_fake_A_images)))
        print("FID VAE_Real_Fake_A_"+eps+": ", fid(np.array(vae_real_A_images), np.array(vae_fake_A_images)))

    # Plot the scores as a line plot and box and whisker plot 
    figs = (12, 8)
    figure(figsize=figs, dpi=100)
    
    plt.boxplot(box_plot_vals, labels = box_plot_names, showmeans=True)
    plt.xticks(rotation=90)
    plt.savefig("SSIM_A_Boxplot_epoch.png")
    plt.show()

individual_scores()
per_epoch()
