import os
import cPickle
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 24


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def readimage1(path):
    with open(path, 'rb') as fo:
        dict = cPickle.load(fo)
        for key in dict.keys():
            yield key

def readimage(path):
    fp = open(path, "rb")
    dict = cPickle.load(fp)
    for i in range(len(dict["labels"])):
        image = dict["labels"][i]
        label = dict["data"][i]
        yield image, label

# Make sure to download and extract CIFAR-10 data before
# running this (https://www.cs.toronto.edu/~kriz/cifar.html)

def distored(images):
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    print images.shape
    for i in range(images.shape[0]):
        feature = images[i,:]
        distorted_image = tf.random_crop(feature, [height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
        float_image = tf.image.per_image_standardization(distorted_image)
        #float_image.set_shape([height, width, 3])
        print float_image
        images[i, :] = float_image
    return images

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = cPickle.load(f)
    images = datadict['data']
    labels = datadict['labels']
    images = images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
    images = distored(images)
    print "image shape=", images.shape
    labels = np.array(labels)
    return images, labels

def load_CIFAR10(ROOT):
    """ load all of cifar """
    images_list = []
    labels_list = []
    for index in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (index, ))
        images, labels = load_CIFAR_batch(f)
        images_list.append(images)
        labels_list.append(labels)
    train_images = np.concatenate(images_list)
    train_labels = np.concatenate(labels_list)
    del images_list, labels_list
    test_images, test_labels = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    path = "../data/cifar-10-batches-py/"
    train_images, train_labels, test_images, test_labels = load_CIFAR10(path)
    print len(train_images)
    print len(train_labels)