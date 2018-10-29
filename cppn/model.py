import tensorflow as tf
import numpy      as np
import math

from collections import namedtuple


Config = namedtuple("Config", 
                    [ "net_size"
                    , "input_size"
                    , "z_dim"
                    , "activations"
                    , "colours"
                    , "norms"
                    ])

Model = namedtuple("Model",
                    [ "z"
                    , "xs" # Input
                    , "ys" # Output
                    , "loss"
                    , "to_match"
                    ])


def net (config, xs):
    print(" Defining a net.")
    print("  xs:", xs)

    h    = xs
    init = tf.random_normal_initializer(mean=0, stddev=1, dtype=tf.float64)

    for func in config.activations:
        h = tf.layers.dense( h
                           , config.net_size
                           , activation         = func
                           , kernel_initializer = init
                           , bias_initializer   = init
                           )
        print("  h:", h)

    ys   = tf.layers.dense(h, config.colours, activation=None)
    return ys


def build_model (config, height, width, reset=True):
    if reset:
        tf.reset_default_graph()

    batch_size = 1 # We are specifically designed for one batch.
    pixels     = width*height
    init       = tf.random_normal_initializer(mean=0, stddev=1)
    coord_dims = config.input_size - config.z_dim

    xs         = tf.placeholder(tf.float32, shape = [ 1, None, coord_dims     ])
    to_match   = tf.placeholder(tf.float32, shape = [ 1, None, config.colours ])

    z_val = np.random.normal(0, 1, size=config.z_dim)
    z     = tf.Variable(z_val, dtype=tf.float32, trainable=False)

    pixel_ones = tf.ones([1, tf.shape(xs)[1], 1])
    h          = tf.concat([xs, pixel_ones * z], axis = 2)

    ys   = tf.map_fn(lambda xs: net(config, xs), h)
    ys   = tf.nn.sigmoid(ys)
    loss = tf.losses.mean_squared_error(labels=to_match, predictions=ys)

    # TODO: Because of the dependence on width/height here, this is wrong,
    #       because actually it's dynamic.
    tf.summary.image("ys",       tf.reshape(ys,       [batch_size, height, width, config.colours]))
    tf.summary.image("to_match", tf.reshape(to_match, [batch_size, height, width, config.colours]))
    tf.summary.scalar("loss", loss)
    
    model = Model( xs       = xs
                 , ys       = ys
                 , z        = z
                 , loss     = loss
                 , to_match = to_match
                 )

    return model


def get_input_data (config, height, width, start, end):

    print("Getting input data....")
    print("  config:", config)
    print("  height:", height)
    print("   width:", width)
    print("   start:", start)
    print("     end:", end)

    x = np.linspace(start, end, num = width)
    y = np.linspace(start, end, num = height)
    return get_input_data_(config, x, y)


def get_input_data_ (config, x, y):
    xx, yy = np.meshgrid(x, y)
    zz     = [ f([xx, yy]).ravel() for f in config.norms ]
    r      = np.array([ xx.ravel(), yy.ravel()] + zz )
    return np.transpose(r)


def stitch_together (yss, rows, columns):
    """ Given that we had to compute the things separately, let's stitch them
        together.

        We know that our loop builds things like so:

        yss = [a, b, c, d]

        image (512x512)
             = 
                 b | d
                 -----
                 a | c

        yss = [a, b, c, d, e, f, g, h, i, j]
        image (768x768)
             = 

                 c | f | i
                 ---------
                 b | e | h
                 ---------
                 a | d | g

        So then plan is just to concat along rows first,
        then concat the rows horiztonally.
    """


    result = []

    # axis 0 = height
    # axis 1 = width

    for c in range(columns):
        elts = yss[c*rows:(c+1)*rows]
        elts = np.concatenate(elts, axis=0)
        result.append(elts)

    result = np.concatenate(result, axis=1)
    return result


def forward (sess, config, model, z, height, width, border=0):
    max_size = 150
    start    = -1 - border
    end      = 1 + border

    if (width * height) > (max_size * max_size):
        # We just want to call "get_input_data_" with a subset
        # of x's and y's.
        sw    = math.ceil(width  / max_size)
        sh    = math.ceil(height / max_size)
        x     = np.linspace(start, end, num = width)
        y     = np.linspace(start, end, num = height)

        results   = []

        for i in range(sw):
            for j in range(sh):
                start_x = i * max_size
                start_y = j * max_size

                end_x = start_x + min(max_size, width  - start_x)
                end_y = start_y + min(max_size, height - start_y)
                
                section_x = x[start_x:end_x]
                section_y = y[start_y:end_y]

                xs = get_input_data_(config, section_x, section_y)
                ys = forward_(sess, config, model, z, section_y.shape, section_x.shape, xs)

                ys = np.reshape(ys, 
                        (section_y.shape[0], section_x.shape[0], config.colours))

                results.append(ys)
     
        return stitch_together(results, sh, sw)
    else:
        xs = get_input_data(config, height, width, start, end)
        ys = forward_(sess, config, model, z, height, width, xs)
        ys = np.reshape(ys, (height, width, config.colours))
        return ys


def forward_ (sess, config, model, z, height, width, xs):
    xss = [xs] # Note; we bundle up in a list here. Because of how
    ys = sess.run( model.ys, feed_dict = { model.z: z, model.xs: xss } )
    return ys
