from collections import namedtuple

import tensorflow as tf
import numpy      as np
import cppn.model as cppn_model


FullModel = namedtuple("FullModel",
                       [ "vae"
                       , "gen_images"
                       , "discrim_loss"
                       , "vae_loss"
                       ])


VaeModel = namedtuple("VaeModel",
                      [ "z_mean"
                      , "z_log_sigma_sq"
                      ])



def build_model (config, height, width, images, reset=True):
    if reset: 
        tf.reset_default_graph()

    print("Building the full model!")
    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    # >> VAE >>
    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    z_dim = 10
    vae   = build_vae(config, height, width, images, z_dim=z_dim)

    print("vae.z_mean:        ", vae.z_mean)
    print("vae.z_log_sigma_sq:", vae.z_log_sigma_sq)
    #
    # In the vae, we have:
    #   vae.z_mean.shape         = [batch_size, z_dim]
    #   vae.z_log_sigma_sq.shape = [batch_size, z_dim]

    eps    = tf.random_normal(tf.shape(vae.z_mean), 0, 1, dtype=tf.float64)
    stddev = tf.multiply(tf.sqrt(tf.exp(vae.z_log_sigma_sq)), eps)
    zs     = tf.add(vae.z_mean, stddev)
    # Now:
    #   zs.shape         = [batch_size, z_dim]
    #   vae.images.shape = [batch_size, height, width, colours]
 
    print("zs:", zs)


    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    #   >> Generator >>
    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    images_shape = tf.shape(images)

    batch_size = images_shape[0]
    colours    = 3
    pixels     = width*height

    start = -1
    end   =  1

    pixel_input = tf.map_fn(lambda _: cppn_model.get_input_data(config, height, width, start, end)
                           , images
                           )

    print("pixel_input:", pixel_input)

    # Now:
    #   pixel_input.shape = [batch_size, pixels, coords]
    
    
    # We have:
    #   pixel_input.shape = [batch_size, pixels, coords]
    #
    # So, now we need:
    #   xs.shape          = [batch_size, pixels, coords + z_dim]
    #
    # where we add on the "zs". But, recall that:
    #
    #   zs.shape         = [batch_size, z_dim]
    #
    # so we need to bump this up first, and _then_ concat.

    def f(z):
        pixel_ones = tf.ones([pixels, 1], dtype=tf.float64)
        return pixel_ones * z

    xs = tf.map_fn(lambda z: f(z), zs)
    print("xs:", xs)
    # Now:
    #   xs.shape = [batch_size, pixels, z_dim]
    
    xs = tf.concat([pixel_input, xs], axis = 2)
    print("xs:", xs)
    # Now:
    #   xs.shape = [batch_size, pixels, colours + z_dim]

    gen_images = tf.map_fn(lambda xs: cppn_model.net(config, xs), xs)
    print("gen_images:", gen_images)
    # Now:
    #   ys.shape = [batch_size, pixels, colours]

    gen_images = tf.reshape(gen_images, [batch_size, height, width, config.colours])
    print("gen_images:", gen_images)


    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    # >> Discriminator >>
    #
    # <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- -> <- ->
    #
    # => Relativistic-GAN Losses
    y_real = build_discriminator(config, height, width, images)
    y_fake = build_discriminator(config, height, width, gen_images, reuse=True)
    y      = tf.ones_like(y_real)

    discrim_loss = ( tf.reduce_mean((y_real - tf.reduce_mean(y_fake) - y) ** 2)
                   + tf.reduce_mean((y_fake - tf.reduce_mean(y_real) + y) ** 2)
                   ) / 2


    # => VAE Losses
    flat_images     = tf.layers.flatten(images)
    flat_gen_images = tf.layers.flatten(gen_images)

    reconstr_loss   = -tf.reduce_sum(flat_images * tf.log(1e-10 + flat_gen_images)
                       + (1 - flat_images) * tf.log(1e-10 + 1 - flat_gen_images), 1)

    latent_loss     = -0.5 * tf.reduce_sum(1 + vae.z_log_sigma_sq
                                       - tf.square(vae.z_mean)
                                       - tf.exp(vae.z_log_sigma_sq), 1)

    reconstr_loss_m = tf.reduce_mean(reconstr_loss)  / pixels
    latent_loss_m   = tf.reduce_mean(reconstr_loss)  / pixels
    vae_loss        = reconstr_loss_m + latent_loss_m

    model = FullModel( vae          = vae
                     , gen_images   = gen_images
                     , discrim_loss = discrim_loss
                     , vae_loss     = vae_loss
                     )


    tf.summary.image("in_images",       images)
    tf.summary.image("gen_images",      gen_images)
    tf.summary.scalar("vae_loss",       vae_loss)
    tf.summary.scalar("reconstr_loss",  reconstr_loss_m)
    tf.summary.scalar("latent_loss",    latent_loss_m)
    tf.summary.scalar("discrim_loss",   discrim_loss)
    tf.summary.scalar("total_loss",     vae_loss + discrim_loss)

    return model


def build_vae (config, width, height, images, vae_size=128, z_dim=10):
    print("Building the VAE ...")

    with tf.variable_scope("vae"):
        print("  images", images.shape)
        flatten = tf.layers.flatten(images)

        h1 = tf.layers.dense(flatten, vae_size)
        h2 = tf.layers.dense(h1,      vae_size)

        z_mean         = tf.layers.dense(h2, z_dim, activation=tf.nn.tanh)
        z_log_sigma_sq = tf.layers.dense(h2, z_dim, activation=tf.nn.tanh)

        vae = VaeModel( z_mean         = z_mean
                    , z_log_sigma_sq = z_log_sigma_sq
                    )

        return vae


def build_discriminator (config, width, height, images, reuse=False):
    print("Building the discriminator...")

    with tf.variable_scope("discrim", reuse=reuse): 
        assert width == height, "Only squares for now."

        d_h_size = 128
        padding  = "same"

        def conv (x, features, strides=2):
            c = tf.layers.conv2d(x, features, 
                    kernel_size=4, 
                    strides=strides, 
                    use_bias=False,
                    padding=padding)
            return c

        c1 = conv(images, d_h_size)
        print("  c1", c1)

        hidden = tf.nn.selu(c1)

        size = width // 2
        mult = 1

        while size > 4:
            print("  hidden", hidden)

            hidden = conv(hidden, d_h_size * 2 * mult)
            hidden = tf.nn.selu(hidden)

            size  = size // 2
            mult *= 2

        final = conv(hidden, 1, strides=1)
        final = tf.squeeze(final)

        return final




