import tensorflow as tf

def fugu (x, f, g=lambda x: x, point=0):
    cond   = tf.less(x, point)
    return tf.where(cond, f(x), g(x))

# Map of activation functions. Note that you don't want to edit this once
# it's been run, because the name is saved in the parameter config file.
AFM = { "tanh":                 tf.nn.tanh
      , "selu":                 tf.nn.selu
      , "relu":                 tf.nn.relu
      , "softplus":             tf.nn.softplus
      , "fugu-tanh":            lambda x: fugu(x, tf.tanh)
      # Penalised-Tanh
      , "ptanh":                lambda x: fugu(x, tf.tanh, lambda x1: 0.25 * tf.tanh(x1))
      , "fugu-sigmoid":         lambda x: fugu(x, tf.nn.sigmoid)
      , "fugu-double-sigmoid":  lambda x: fugu(x, tf.nn.sigmoid, lambda x: 1 - tf.nn.sigmoid(x))
      , "fugu-dropout-sigmoid": lambda x: fugu(x, lambda x1: tf.nn.dropout(x1, 0.5), lambda x: 1 - tf.nn.sigmoid(x))
      }

