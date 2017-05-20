import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 32
HEIGHT = 32


def genTFrecord(dir, classes,destpath):
    writer= tf.python_io.TFRecordWriter(destpath)
    for index,name in enumerate(classes):
        path=dir+name+'/'
        for img_name in os.listdir(path):
            img_path=path+img_name
            image=Image.open(img_path)
            image = image.resize((WIDTH,HEIGHT))
            img = np.asarray(image)
            print img.shape,img_path
            #img = np.transpose(img, [1, 0, 2])
            #print img.shape[0],img.shape[1],img.shape[2]
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[2]])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def tfcord2jpg(source_list, save_dir):
    filename_queue = tf.train.string_input_producer(source_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)


    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            img_eval, lable_eval,width_eval,height_eval,depth_eval = sess.run([image, label, width, height, depth])
            print width_eval, height_eval, depth_eval
            img_eval = np.reshape(img_eval,[width_eval, height_eval, depth_eval])
            img = Image.fromarray(img_eval, 'RGB')
            label_eval = label.eval()
            img.save(save_dir + str(i) + '_''Label_' + str(label_eval) + '.jpg')
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main1__":
    dir = "../../data/re/train/"
    classes = ['3','4','5','6','7']
    destpath = "../../data/train.tfrecords"
    genTFrecord(dir,classes,destpath)

if __name__ == "__main__":
    source_list = ["../../data/train.tfrecords"]
    save_dir = "../../data/jpg/"
    tfcord2jpg(source_list, save_dir)
