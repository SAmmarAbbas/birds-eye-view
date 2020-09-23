import tensorflow as tf

slim = tf.contrib.slim


def cnn_vggm(inputs, num_classes, model, weight_decay=5e-4, reuse=None):
    regularizer = slim.l2_regularizer(weight_decay)

    with tf.variable_scope('vgg_m', 'vgg_m', [inputs], reuse=reuse) as sc:
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=regularizer):
            # First Group: Convolution + Pooling
            net = slim.conv2d(inputs, 96, [7, 7], stride=[2, 2], padding='VALID', scope='conv1')  # 28x28x20
            net = tf.nn.local_response_normalization(net, depth_radius=model[1]['param'][0], bias=model[1]['param'][1],
                                                     alpha=model[1]['param'][2], beta=model[1]['param'][3],
                                                     name='norm1')
            net = tf.pad(net, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.max_pool2d(net, [3, 3], stride=[2, 2], scope='pool1')  # missing padding here [0, 1, 0, 1]
            print(net.shape)

            # Second Group: Convolution + Pooling
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.conv2d(net, 256, [5, 5], stride=[2, 2], padding='VALID', scope='conv2')  # missing padding
            print(net.shape)
            net = tf.nn.local_response_normalization(net, depth_radius=model[4]['param'][0], bias=model[4]['param'][1],
                                                     alpha=model[4]['param'][2], beta=model[4]['param'][3],
                                                     name='norm2')
            net = tf.pad(net, [[0, 0], [0, 1], [0, 1], [0, 0]],
                         "CONSTANT")  # batches, height, width, channels (inside t,b,l,r)
            net = slim.max_pool2d(net, [3, 3], stride=[2, 2],
                                  scope='pool2')  # ## missing padding here [0, 1, 0, 1] (t, b, l, r)
            print(net.shape)

            # Third Group: Convolution
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.conv2d(net, 512, [3, 3], stride=[1, 1], padding='VALID', scope='conv3')  # missing padding
            print(net.shape)

            # Fourth Group: Convolution
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.conv2d(net, 512, [3, 3], stride=[1, 1], padding='VALID', scope='conv4')  # missing padding
            print(net.shape)

            # Fifth Group: Convolution + Pooling
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.conv2d(net, 512, [3, 3], stride=[1, 1], padding='VALID', scope='conv5')  # missing padding
            net = tf.pad(net, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")  # batches, height, width, channels
            net = slim.max_pool2d(net, [3, 3], stride=[2, 2], scope='pool5')  # ## missing padding here [0, 1, 0, 1]
            print(net.shape)

            # Sixth Group: Fully Connected
            net = slim.conv2d(net, 4096, [6, 6], padding='VALID', scope='fc6')
            print(net.shape)

            # Seventh Group: Fully Connected
            net = slim.conv2d(net, 4096, [1, 1], padding='VALID', scope='fc7')
            print(net.shape)

            # Eigth Group: Fully Connected
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, padding='VALID', scope='fc8')
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            print(net.shape)

    return net
