import tensorflow as tf
import numpy      as np
import math

from collections                    import namedtuple
from tensorflow.python.keras        import Sequential
from tensorflow.python.keras.layers import Dense


def build_model (height, width):
    tf.reset_default_graph()
    # 1. Resize images
    # 2. Convert to between 0,1
    # 3. Normalise to mean 0.5 and standard deviation 0.5
    pass


def discriminator (width, height):
    assert width == height, "Only squares for now."

    colours  = 3
    d_h_size = 10

    images = tf.placeholder(tf.float32, shape = [ None, height, width, coord_dims ])

    c1 = tf.layers.conv2d(images, d_h_size, kernel_size=4, strides=2,
            use_bias=False)
    hidden = tf.nn.selu(cc)

    size = width // 2
    mult = 1

    while size > 4:
        hidden = tf.layers.conv2d(hidden, d_h_size * 2 * mult,
                kernel_size=4,
                strides=2,
                use_bias=False)
        hidden = tf.nn.selu(hidden)
        size = size // 2
        mult *= 2

    final = tf.layers.conv2d(hidden, 1, kernel_size=4, strides=1, use_bias=False)
    final = tf.squeeze(final)
    final = tf.nn.sigmoid(final)

    # Do loss stuff

    return final
