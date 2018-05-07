import tensorflow as tf


def conv_layer(name, X, in_channels, out_filters, ksize, stride, padding="SAME", trainable=True,
               with_elu=True,
               with_bn=True, momentum=0.99, epsilon=1e-5):
    if not hasattr(ksize, "__len__"):
        ksize = [ksize, ksize]
    if not hasattr(stride, "__len__"):
        stride = [stride, stride]
    with tf.variable_scope(name):
        w = tf.get_variable("weight", [ksize[0], ksize[1], in_channels, out_filters], tf.float32,
                            # tf.contrib.layers.xavier_initializer(),
                            tf.truncated_normal_initializer(0.0, 0.01),
                            trainable=trainable)
        b = tf.get_variable("bias", [out_filters], tf.float32,
                            # tf.contrib.layers.xavier_initializer(),
                            tf.constant_initializer(0.0),
                            trainable=trainable)
        Y = tf.add(tf.nn.conv2d(X, w, [1, stride[0], stride[1], 1], padding), b)
        if with_bn:
            Y = tf.layers.batch_normalization(Y, momentum=momentum, epsilon=epsilon,
                                             center=True, scale=True, renorm=True,
                                             trainable=True,
                                             training=trainable)
        if with_elu:
            Y = tf.nn.elu(Y)
    return Y, w, b


def fc_layer(name, X, in_channels, out_filters, trainable=True,
             with_elu=True, 
             with_bn=True, momentum=0.99, epsilon=1e-5):
    with tf.variable_scope(name):
        w = tf.get_variable("weight", [in_channels, out_filters], tf.float32,
                            # tf.contrib.layers.xavier_initializer(),
                            tf.truncated_normal_initializer(0.0, 0.01),
                            trainable=trainable)
        b = tf.get_variable("bias", [out_filters], tf.float32,
                            # tf.contrib.layers.xavier_initializer(),
                            tf.constant_initializer(0.0),
                            trainable=trainable)
        Y = tf.add(tf.matmul(tf.reshape(X, [-1, in_channels]), w), b)
        if with_bn:
            Y = tf.layers.batch_normalization(Y, momentum=momentum, epsilon=epsilon,
                                             center=True, scale=True, renorm=True,
                                             trainable=True,
                                             training=trainable)
        if with_elu:
            Y = tf.nn.elu(Y)
    return Y, w, b


def lstm_layer(name, X, in_channels, out_filters):
    with tf.variable_scope(name):
        step_size = tf.placeholder(tf.float32, [1])
        cell = tf.nn.rnn_cell.BasicLSTMCell(out_filters, state_is_tuple=True)

        c = tf.placeholder(tf.float32, [1, cell.state_size.c])
        h = tf.placeholder(tf.float32, [1, cell.state_size.h])
        init_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        Y, state = tf.nn.dynamic_rnn(cell,
                                     tf.reshape(X, [1, -1, in_channels]),
                                     initial_state=init_state,
                                     sequence_length=step_size,
                                     time_major=False)
    return Y, (state, (c, h), step_size)


def max_pool(name, X, ksize, padding="SAME"):
    with tf.variable_scope(name):
        Y = tf.nn.max_pool(X, [1, ksize, ksize, 1], [1, ksize, ksize, 1], padding)
    return Y
