import cppn.model                   as model
import cppn.relativistic_gan_model  as gan_model

from cppn.data import make_dataset

import os
import scipy.misc as sp
import tensorflow as tf
import numpy      as np
import pickle
import json
import uuid
from tensorflow.core.protobuf import config_pb2

def make_session (server):
    config = config_pb2.ConfigProto(isolate_session_state=True)
    return tf.Session(server, config=config) 


def generate (z, height, width, config, out, server):
    """ Just randomly generate a CPPN-style image.
    """
    net = model.build_model(config, height, width)

    with make_session(server) as sess:
        sess.run(tf.global_variables_initializer())
        ys = model.forward(sess, config, net, z, height, width)

    print(f"z = {z}")
    sp.imsave(out, ys)



def sample (config, checkpoint_dir, height, width, server, out, z, z_steps, border, border_steps):
    """ Sample from a pre-trained model and emit an image.
    """
    with open(f"{checkpoint_dir}/config.pkl", "rb") as f:
        config_data = pickle.load(f)

    if z is not None:
        if z is str:
            z = np.array( [ float(n) for n in z.split(',') ] )
    else:
        z = config_data["z"]
    
    net = model.build_model(config, height, width)

    if z_steps > 0:
        # interpolate from z to zn
        zn = np.random.normal(-1, 1, size=config.z_dim)
        # zs = np.

        z_ = lambda t: t * z + (1 - t) * zn
        zs = [z_(t) for t in np.linspace(0, 1, z_steps)]


    with make_session(server) as sess:
        saver       = tf.train.Saver()
        checkpoint  = tf.train.latest_checkpoint(checkpoint_dir)
        ckpt_number = int(os.path.basename(checkpoint).split('.')[0])

        config_data["checkpoint"] = ckpt_number

        saver.restore(sess, checkpoint)

        if z_steps > 0:
            frames = [ model.forward(sess, config, net, z_, height, width, border) for z_ in zs ]
        else:
            if border_steps > 0:
                borders = np.linspace(0, border, num = border_steps)
                frames  = [ model.forward(sess, config, net, z, height, width, b) for b in borders ]
                frames  = reversed(frames)
            else:
                ys = model.forward(sess, config, net, z, height, width, border=border)

    with open(f"{out}.json", "w") as f:
        config_data["z"]              = config_data["z"].tolist()
        config_data["checkpoint_dir"] = checkpoint_dir
        json.dump(config_data, f)

    if z_steps > 0 or border_steps > 0:
        for k, ys in enumerate(frames):
            sp.imsave(out + "/%05d.png" % k, ys)
    else:
        sp.imsave(out, ys)


def train_gan (ctx, directory, server, config, base_log_dir="logs", log_directory=None):
    """ Train a CPPN on a given directory.
    """
    height     = 32
    width      = 32
    epochs     = 50
    batch_size = 1
    lr         = 0.001
    log_every  = 1000

    dataset    = make_dataset(directory, height, width)
    dataset    = dataset.repeat(epochs)
    dataset    = dataset.batch(batch_size)
    iterator   = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    model = gan_model.build_model(config, height, width, next_batch, reset=False)

    optim      = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9)
    loss_op    = model.vae_loss + model.discrim_loss
    train_step = optim.minimize(loss_op)

    merged   = tf.summary.merge_all()

    if log_directory is None:
        id_           = str(uuid.uuid1())[:8]
        summaries_dir = f"{base_log_dir}/{id_}"
    else:
        summaries_dir = log_directory

    saver  = tf.train.Saver()

    print(f"logs: {summaries_dir}")

    k = 0
    with tf.train.MonitoredTrainingSession() as sess:
        print("Training ... ")
        writer = tf.summary.FileWriter(f"{summaries_dir}/train", sess.graph)

        while not sess.should_stop():
            _, loss = sess.run( [ train_step, loss_op ] )
            if k % log_every == 0:
                summaries = sess.run(merged)
                writer.add_summary(summaries, k)
                print(f"{k}:  loss:", loss)
            k += 1

    print("Done!")



def train_matching (ctx, image, z, server, config, lr, steps, log_every, base_log_dir, log_directory=None):
    """ Train a model to match the given input image.
    """
    loaded = sp.imread(image, mode="RGB") / 255.0

    (height, width, colours) = loaded.shape

    samples_w = 50
    ratio     = samples_w / width
    samples_h = int(ratio * height)

    start = -1
    end   =  1

    net     = model.build_model(config, samples_h, samples_w)
    xs      = model.get_input_data(config, height, width, start, end)

    optim      = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.6, beta2=0.9)
    train_step = optim.minimize(net.loss)

    to_match  = loaded.reshape( (-1, colours) )
    to_match_ = sp.imresize(loaded, (samples_h, samples_w, colours)).reshape((-1, colours))

    img_xs   = model.get_input_data(config, samples_h, samples_w, start, end)
    merged   = tf.summary.merge_all()

    if log_directory is None:
        id_           = str(uuid.uuid1())[:8]
        summaries_dir = f"{base_log_dir}/{id_}"
    else:
        summaries_dir = log_directory

    with make_session(server) as sess:
        print(f"Summaries dir: {summaries_dir}")
        writer = tf.summary.FileWriter(f"{summaries_dir}/train", sess.graph)
        saver  = tf.train.Saver()
        start  = 0

        if log_directory is not None:
            checkpoint = tf.train.latest_checkpoint(summaries_dir)
            print(f"Reloading checkpoint: {checkpoint}")
            start = int(os.path.basename(checkpoint).split('.')[0])
            saver.restore(sess, checkpoint)
        else:
            sess.run(tf.global_variables_initializer())
            with open(f"{summaries_dir}/config.pkl", "wb") as f:
                # Get the params of the parent, which defines the model.
                data = {}
                data["params"] = ctx.parent.params
                data["z"]      = z
                data["lr"]     = lr
                data["image"]  = image
                pickle.dump(data, f)

        loss = None
        for k in range(start, steps):
            b = np.random.randint( height*width, size=(samples_w * samples_h) )

            data = { net.to_match: to_match[b]
                   , net.xs      : xs[b]
                   , net.z       : z
                   }

            _, loss = sess.run( [ train_step, net.loss ], feed_dict=data )

            if k % log_every == 0 or (k == steps - 1):
                print(f"Step {k}, Loss: {loss}")

                data[net.xs]       = img_xs
                data[net.to_match] = to_match_

                summaries = sess.run(merged, feed_dict=data)
                writer.add_summary(summaries, k)
                saver.save(sess, f"{summaries_dir}/{k}.ckpt")

    # Update the loss in the pickle, if we can.
    if loss is not None:
        with open(f"{summaries_dir}/config.pkl", "rb") as f:
            data = pickle.load(f)
            data["loss"] = float(loss)

        with open(f"{summaries_dir}/config.pkl", "wb") as f:
            pickle.dump(data, f)

        with open("./result.json", "w") as f:
            json.dump({"loss": float(loss)}, f)

    return summaries_dir


