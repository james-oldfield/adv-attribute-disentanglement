import tensorflow as tf


# Based on: https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow/blob/master/train_celeba_wgan_gp.py
def gradient_penalty(real, fake, f):
    def interpolate(a, b):
        shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.get_shape().as_list())
        return inter

    x = interpolate(real, fake)
    pred = f(x, reuse=True)
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp