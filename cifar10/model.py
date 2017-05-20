import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from input import  *

tf.logging.set_verbosity(tf.logging.INFO)


# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100, "")
tf.app.flags.DEFINE_integer('num_classes', 10, "")

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

def inference(images):
    #input_layer = tf.reshape(images, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=images,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool1 =tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

    conv2 = tf.layers.conv2d(inputs=norm1,
                             filters=64,
                             kernel_size=[5,5],
                             padding="same",
                             activation=tf.nn.relu)

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    pool2 = tf.layers.max_pooling2d(inputs=norm2,
                                    pool_size=[2,2],
                                    strides=2)

    #pool2_flat = tf.reshape(pool2, [FLAGS.batch_size, -1])
    pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * 64])

    dense3 = tf.layers.dense(inputs=pool2_flat, units=384, activation=tf.nn.relu)

    dense4 = tf.layers.dense(inputs=dense3,
                             units=192,
                             activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense4, units=FLAGS.num_classes)

    return logits

def train(logits,labels):
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=FLAGS.num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD"
    )
    return train_op

'''
if __name__ == "__main__":
    data_dir = "../data/cifar-10-batches-bin"
    save_dir = "../model/cifar_model/"
    batch_size = 100
    images, labels = distorted_inputs(data_dir, batch_size)
    logits = inference(images)
    #loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=FLAGS.num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    loss_mean = tf.reduce_mean(loss)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD"
    )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        tf.train.start_queue_runners(sess=sess)
        count = 0
        while count <= 100000:
            count += 1
            _, loss_val = sess.run([train_op, loss_mean])
            if count % 100 == 0:
                print "count=",count,loss_val
            if count % 10000 == 0:
                checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=count)
'''


def accuracy(predictions, labels):
    pred_class = np.argmax(predictions, 1)
    return (100.0 * np.sum(pred_class == labels) / predictions.shape[0])

if __name__ == "__main__":
    data_dir = "../data/cifar-10-batches-bin"
    model_dir = "../model/cifar_model/"
    batch_size = 100
    images, labels = inputs(False, data_dir, batch_size)
    logits = inference(images)
    init_op = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(100):
                predict, classes = sess.run([logits,labels])
                print accuracy(predict, classes)
        coord.request_stop()
        coord.join(threads)
