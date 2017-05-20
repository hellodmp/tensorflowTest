import tensorflow as tf

if __name__ == "__main1__":
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                          name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")
    w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        save_path = saver.save(sess, "./model.ckpt")
        print weights
        print w_twice
        print biases
    print "ok"


if __name__ == "__main2__":
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a * b

    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)
    print(node1, node2)
    sess = tf.Session()
    print(sess.run(adder_node, {a: 3, b:4.5}))



if __name__ == "__main3__":
    W = tf.Variable([0.3], tf.float32)
    b = tf.Variable([-0.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    init = tf.global_variables_initializer()

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    x_train = [1,2,3,4]
    y_train = [0,-1,-2,-3]

    sess = tf.Session()
    sess.run(init)
    print(sess.run(loss, {x:x_train, y:y_train}))

    for i in range(1000):
        sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    print(sess.run([W, b]))

if __name__ == "__main__":
    #x = tf.placeholder(tf.float32, [1, 7])
    x = tf.Variable(tf.ones([2,7]))
    w = tf.Variable(tf.ones([7,10]))
    b = tf.Variable(tf.ones([10]))
    value = tf.matmul(x, w) + b
    y = tf.nn.softmax(tf.matmul(x, w) + b,0)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print sess.run(w)
    print sess.run(b)
    print sess.run(y)
    print sess.run(tf.nn.softmax(w,0))



