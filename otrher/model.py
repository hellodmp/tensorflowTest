from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from cifar10.input import *

tf.logging.set_verbosity(tf.logging.INFO)


# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100, "")
tf.app.flags.DEFINE_integer('num_classes', 10, "")

FLAGS = tf.app.flags.FLAGS

def inference(images,labels, mode):
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

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=FLAGS.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD"
        )
    predictions = {
        "classes":tf.arg_max(input=logits, dimension=1),
        "probability":tf.nn.softmax(logits, name="softmax_tensor")
    }

    return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def input_fn():
    data_dir = "../data/cifar-10-batches-bin"
    batch_size = 100
    images, labels = distorted_inputs(data_dir, batch_size)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        return images, labels
        coord.request_stop()
        coord.join(threads)

def input_test_fn():
    data_dir = "../data/cifar-10-batches-bin"
    batch_size = 100
    images, labels = inputs(True, data_dir, batch_size)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        return images, labels
        coord.request_stop()
        coord.join(threads)

'''
def main(unused_argv):
    path = "../data/cifar-10-batches-py/"
    #train_images, train_labels, test_images, test_labels = load_CIFAR10(path)
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log = {}
    classifier = learn.Estimator(model_fn=inference, model_dir="../model/cifar_model")
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    classifier.fit(
        input_fn=input_fn,
        steps=10000)

    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    #Evaluate the model and print results
    test_images,test_labels = input_test_fn()
    eval_results = classifier.evaluate(x=test_images, y=test_labels, metrics=metrics)
    print(eval_results)
    print "ok"

'''
def main1(unused_argv):
    path = "../model/cifar-model/"
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }
    with tf.Graph().as_default() as g:
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session(graph=g) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
        print "ok"


def main(unused_argv):
    data_dir = "../data/cifar-10-batches-bin"
    batch_size = 100
    images, labels = inputs(True, data_dir, batch_size)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tensors_to_log = {}
        classifier = learn.Estimator(model_fn=inference, model_dir="../model/cifar_model")
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        images, labels = distorted_inputs(data_dir, batch_size)

        classifier.fit(
            x=images,
            y=labels,
            steps=10000)

        coord.request_stop()
        coord.join(threads)



if __name__ == "__main__":
    tf.app.run()

