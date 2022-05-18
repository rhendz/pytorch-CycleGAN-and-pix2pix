# Training CycleGAN using AE-based Initialization

# Usage
Please use *ae_cyclegan.ipynb* - this is the primary notebook for this project.

The notebook first clones the augmented cyclegan git repository. Please see [CycleGan Augmentations] for specifics.

The notebook changes into the appropriate cloned working directory, and checks for any updates. Make sure when running the notebook the working directory is *pytorch-CycleGAN-and-pix2pix*.

The apple2orange dataset is downloaded via a script located in datasets and is resized by a script that we provide to make model training easier; however, this script can be easily augmented to resize the image to different sizes e.g. 64x64.

Next, the notebook implements an autoencoder based off [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html] with several modifications. Specifically, model ooptions for the CycleGAN are constructed designated the batch_size, image size, number of channels, preprocess, etc. Next, the train and test datasets are loaded using CycleGAN. Additionally, we insert a custom resnet_3block generator into the Decoder model - we describe this in the attached paper.

This model is then trained and displayed on tensorboard, where loss and val_loss can be monitored along with reconstructed images.

Finally, these trained models are copied (to 'models/vae-gen(A|B)-model.pt') and loaded into CycleGAN via the CycleGAN augmentation mentioned below. The CycleGAN is trained with both VAE and no VAE options and tested with images shown at the end.

# CycleGAN Augmentations
Within models/cycle_gan_model.py constructor, we load the trained decoder models into generator's A and B e.g. apple and oranges.

Additionally, we provide a base option located under options to train CycleGAN model with 'vae' initialized weights.

Finally, we introduce a --netG option for resnet_3blocks instead of resnet_6blocks and resnet_9blocks for quicker training. Please modify this if you would like to use CycleGAN's original block count.

# Notes
Please see the attached *final-paper.pdf* for additional details