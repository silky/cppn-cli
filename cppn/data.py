import tensorflow as tf
import numpy      as np
import os

def get_image_files (directory):
    exts = [".jpg", ".png"]
    for f in os.listdir(directory):

        full_path = os.path.join(directory, f)
        ext       = os.path.splitext(f)[1]

        if ext.lower() not in exts:
            continue

        yield full_path


def make_dataset (directory, height=32, width=32):
    """ Reads an image from a file, decodes it into a dense tensor, and resizes it
        to a fixed shape.
    """

    def _parse_function (filename):
        image_string  = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [height, width]) / 255.
  
        # TODO: Normalise the images to mean 0.5 and std-dev 0.5 ?
        image_resized = tf.cast(image_resized, tf.float64)

        return image_resized 

    files     = list(get_image_files(directory))
    filenames = tf.constant(files)
    dataset   = tf.data.Dataset.from_tensor_slices(filenames)
    dataset   = dataset.map(_parse_function)

    return dataset
