# Resizes the apples to oranges images

from PIL import Image
from numpy import asarray
import os

# Pass the directory of the current images and the size to be resized
def resize_images(directory,size):
    dataset_name = directory.split("/")[-1]
    images = []
    parent_dir = './datasets/'
    new_dir = 'apple2orange_resized_'+str(size)
    parent_dir += new_dir+'/'+dataset_name
    if not os.path.exists(parent_dir):
      os.makedirs(parent_dir)
    print(parent_dir)
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        real_fn = f.replace('\\', '/')
        image = Image.open(real_fn)
        # plt.imshow(image)
        # plt.show()
        data = asarray(image)
        images.append(data)
        newsize = (size, size)
        image = image.resize(newsize)
        # plt.imshow(image)
        # plt.show()
        image.save(parent_dir+'/'+filename)
    
    #return np.array(images)
resize_images("./datasets/apple2orange/testA", 28)
resize_images("./datasets/apple2orange/testB", 28)
resize_images("./datasets/apple2orange/trainA", 28)
resize_images("./datasets/apple2orange/trainB", 28)