# ------------------------------
# code for image-to-image translation
# ------------------------------

import tensorflow as tf
import numpy as np

from funcy import partial, compose
from utils import gradient_penalty

relu = tf.nn.relu
lr = partial(tf.nn.leaky_relu, alpha=0.01)
tanh = tf.tanh
conv = tf.layers.conv2d
dense = tf.layers.dense
mean_cross_entropy = compose(tf.reduce_mean, tf.nn.softmax_cross_entropy_with_logits)


class Attribute:
    def __init__(self, name, n_classes=0, batch_size=16, labels=[], idx=[]):
        self.name = name
        self.n_classes = n_classes
        self.idx = idx
        self.labels = labels

        self.uniform = tf.ones((batch_size, self.n_classes)) / self.n_classes


class Model:
    def __init__(self):
        self.graph = tf.Graph()

    def _parse_function(self, filename, *labels):
        image_string = tf.read_file(filename)
        image_decoded = tf.reshape(
            tf.image.resize_images(
                tf.cast(tf.image.decode_jpeg(image_string, channels=3), tf.float32) / 127.5 - 1.,
                [128, 128]
            ), [self.img_size, self.img_size, 3])

        if len(labels) == 0:
            return image_decoded
        else:
            return image_decoded, labels

    def encoder(self, x, name="", reuse=False):
        with self.graph.as_default():
            if not reuse:
                print('Adding encoder \'{}\' to graph...'.format(name))

            with tf.variable_scope('gen/encoder-{}'.format(name), reuse=reuse):
                norm = tf.contrib.layers.instance_norm
                d = 32

                layers = compose(relu, norm, conv)

                out = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
                out = layers(out, d, 7, 1, padding='valid', name='enc-1')
                out = layers(out, d * 2, 3, 2, padding='same', name='enc-2')
                out = layers(out, d * 4, 3, 2, padding='same', name='enc-3')

                return out

    def decoder(self, z, reuse=False):
        if not reuse:
            print('Adding decoder to graph...')

        with tf.variable_scope('gen/decoder', reuse=reuse):
            norm = tf.contrib.layers.instance_norm
            d = (len(self.attributes) + 1) * 32

            def r_pad(x, n=1):
                return tf.pad(x, [[0, 0], [n, n], [n, n], [0, 0]], 'reflect')

            def residule_block(x, dim, scope='res'):
                with tf.variable_scope('res-{}'.format(scope), reuse=reuse):
                    y = relu(norm(conv(r_pad(x), dim, 3, 1, padding='valid', name='c1')))
                    y = norm(conv(r_pad(y), dim, 3, 1, padding='valid', name='c2'))
                    return y + x

            out = z
            for res_i in range(6):
                out = residule_block(out, d * 4, scope='r{}'.format(res_i))

            layers = compose(relu, norm, conv)

            out = tf.image.resize_nearest_neighbor(out, [128, 128])
            out = layers(r_pad(out), d * 2, 3, 2, padding='valid', name='dec-1')
            out = tf.image.resize_nearest_neighbor(out, [256, 256])
            out = layers(r_pad(out), d, 3, 2, padding='valid', name='dec-2')
            out = conv(r_pad(out, 3), 3, 7, 1, padding='valid', name='dec-3')

            return tanh(out)

    def discriminator(self, x, d=64, reuse=False):
        with self.graph.as_default():
            if not reuse:
                print('Adding discrim to graph...')

            def pad(x):
                return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='constant')

            with tf.variable_scope('disc/', reuse=reuse):
                conv = partial(tf.layers.conv2d, padding='valid', strides=2, kernel_size=4)
                conv_s1 = partial(tf.layers.conv2d, padding='valid', strides=1, kernel_size=3)

                layers = compose(lr, conv)

                out = x

                out = layers(pad(out), d, name='block-1')
                out = layers(pad(out), d * 2, name='block-2')
                out = layers(pad(out), d * 4, name='block-3')
                out = layers(pad(out), d * 8, name='block-4')
                out = layers(pad(out), d * 16, name='block-5')
                out = layers(pad(out), d * 32, name='block-6')

                real_logits = conv_s1(pad(out), 1, name='block-7')

                return real_logits

    def classifier(self, x, name, n_classes=0, z=False, reuse=False):
        """
        We define a single instance of the classifer for each attribute to model.
        We jointly train it to classify the training set's images, whilst simultaneously using the learnt representations for disentangling our latent encodings (and for imposing the 'transfer' loss).
        """

        with self.graph.as_default():
            if not reuse:
                print('Adding classifier \'{}\' to graph...'.format(name))

            def pad(x):
                return tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='constant')

            with tf.variable_scope('class/{}'.format(name), reuse=reuse):
                conv = partial(tf.layers.conv2d, padding='valid', strides=2, kernel_size=4)

                layers = compose(relu, conv)

                d = 64
                out = x

                if not z:
                    # These first two conv blocks are only applied if the input is (128x128x3) image, rather than a latent encoding
                    with tf.variable_scope('0'): out = layers(pad(out), d, name='block-1')
                    with tf.variable_scope('1'): out = layers(pad(out), d * 2, name='block-2')

                return tf.reshape(tf.layers.dense(tf.layers.flatten(out), n_classes, name='logits'), [-1, n_classes])

    def adv_loss(self, fake):
        dX_fake = self.discriminator(fake, reuse=True)

        g_loss = -tf.reduce_mean(dX_fake)

        wd = tf.reduce_mean(self.dX_real) - tf.reduce_mean(dX_fake)
        gp = gradient_penalty(self.X, fake, self.discriminator)
        d_loss = -wd + gp * 10.0

        return d_loss, g_loss

    def build_encoders(self, args):
        with self.graph.as_default():
            self.attributes = [Attribute(name=n) for n in args.attribute_names]

            for i, m in enumerate(self.attributes):
                # define encoder for each attribute: X -> z_i
                m.E = partial(self.encoder, name='E_{}'.format(m.name))

            self.E_0 = partial(self.encoder, name='E_0')

    def build_train_graph(self, args, X_train, labels, attribute_domains, X_test):
        self.img_size = args.img_size
        self.attribute_domains = attribute_domains
        self.batch_size = 16

        # ------------------
        # DATASETS + PLACEHOLDERS
        # ------------------

        with self.graph.as_default():
            self.attributes = []

            self.train_set = tf.data.Dataset.from_tensor_slices(tuple([tf.constant(X_train)] + [tf.constant(l) for l in labels])) \
                .map(self._parse_function) \
                .shuffle(10000) \
                .apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)) \
                .make_initializable_iterator()

            self.test_set = tf.data.Dataset.from_tensor_slices((tf.constant(X_test))) \
                .map(self._parse_function) \
                .shuffle(100) \
                .apply(tf.contrib.data.batch_and_drop_remainder(16)) \
                .repeat() \
                .make_one_shot_iterator()

            self.X, self.labels = self.train_set.get_next()
            self.next_X_te = self.test_set.get_next()

            self.LR = args.learning_rate
            self.LR_PH = tf.placeholder(tf.float32, shape=())
            self.epoch_PH = tf.placeholder(tf.float32, shape=())

            # we randomly permute the order of the encodings for each attribute, along the batch dimension
            self.idx = [tf.placeholder(tf.int32, shape=(self.batch_size), name='{}-idx'.format(n)) for n in args.attribute_names]

            self.attributes = [
                Attribute(name=args.attribute_names[i], n_classes=n, batch_size=self.batch_size, labels=self.labels[i], idx=self.idx[i])
                for i, n in enumerate(attribute_domains)
            ]

            # ------------------
            # ENCODE
            # ------------------
            for i, m in enumerate(self.attributes):
                # define encoder for each attribute: X -> z_m
                m.z = self.encoder(self.X, 'E_{}'.format(m.name))

            self.z_0 = self.encoder(self.X, 'E_0')

            # ------------------
            # DECODE
            # ------------------
            self.X_rec = self.decoder(tf.concat([m.z for m in self.attributes] + [self.z_0], axis=-1))  # reconstruction
            self.X_trn = self.decoder(tf.concat([tf.gather(m.z, m.idx) for m in self.attributes] + [self.z_0], axis=-1), reuse=True)  # synthesised output from 'generalisable' process

            # ------------------
            # CLASSIFY
            # ------------------
            self.dX_real = self.discriminator(self.X)

            for i, m in enumerate(self.attributes):
                # define one classifier for each attribute...
                m.cls = partial(self.classifier, name=m.name, n_classes=m.n_classes)
                # ... and corresponding entry point for its latent encodings
                m.cls_z = partial(self.classifier, name=m.name, n_classes=m.n_classes, z=True)

                m.real_cls_loss = mean_cross_entropy(labels=m.labels, logits=m.cls(self.X))  # L^c_cls (train classifier on ground-truth training set images)
                m.fake_cls_loss = mean_cross_entropy(labels=m.labels, logits=m.cls_z(m.z, reuse=True))  # L^x_cls  (imbue z_m with representations pertaining to attribute m)
                m.trn_cls_loss = mean_cross_entropy(labels=tf.gather(m.labels, m.idx), logits=m.cls(self.X_trn, reuse=True))  # L^x'_cls (encourage reps to be generalisable)

            for i, m in enumerate(self.attributes):
                # mean pairwise 'disentangle' loss between each other attribute
                m.dis_loss = tf.reduce_mean([
                    mean_cross_entropy(labels=m_.uniform, logits=m_.cls_z(m.z, reuse=True))
                    for j, m_ in enumerate(self.attributes) if j != i
                ])

            # pairwise dis loss between z_0 and *every* specified attribute
            self.dis_loss_0 = tf.reduce_mean([mean_cross_entropy(labels=m.uniform, logits=m.cls_z(self.z_0, reuse=True)) for m in self.attributes])

            # ------------------
            # LOSS
            # ------------------
            self.recon_loss = 10. * (tf.reduce_mean(tf.abs(self.X - self.X_rec)))

            self.real_cls_loss = tf.reduce_mean([m.real_cls_loss for m in self.attributes])
            self.fake_cls_loss = tf.reduce_mean([m.fake_cls_loss for m in self.attributes])
            self.transfer_loss = tf.reduce_mean([m.trn_cls_loss for m in self.attributes])

            self.dis_loss = tf.reduce_mean([m.dis_loss for m in self.attributes] + [self.dis_loss_0])

            self.D_loss, self.G_loss = self.adv_loss(self.X_trn)
            self.G_loss += self.recon_loss + self.transfer_loss + self.fake_cls_loss + self.dis_loss
            self.C_loss = self.real_cls_loss
            # ==================

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen/')
            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='class/')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim_G = tf.train.AdamOptimizer(self.LR_PH, beta1=args.beta1).minimize(self.G_loss, var_list=G_vars)
                self.optim_D = tf.train.AdamOptimizer(self.LR_PH, beta1=args.beta1).minimize(self.D_loss, var_list=D_vars)
                self.optim_C = tf.train.AdamOptimizer(self.LR_PH, beta1=args.beta1).minimize(self.C_loss, var_list=C_vars)

            # ------------------
            # TRANSFER
            # ------------------

            self.transfers = self.switch(self.X)
            self.stack = tf.squeeze(tf.concat([tf.concat(tf.split(t, self.batch_size), axis=2) for t in self.transfers], axis=1))

            self.sample = tf.clip_by_value(tf.divide(tf.add(self.stack, 1.0), 2.0), 0, 255)
            self.sample = tf.image.convert_image_dtype(self.sample, saturate=True, dtype=tf.uint16)

            self.sample = tf.image.encode_png(self.sample)

            sums = [
                tf.summary.scalar('loss/real_cls_cls', self.real_cls_loss),
                tf.summary.scalar('loss/fake_cls_loss', self.fake_cls_loss),

                tf.summary.scalar('loss/dis_loss', self.dis_loss),
                tf.summary.scalar('loss/trn_loss', self.transfer_loss),

                tf.summary.scalar('loss/recon_loss', self.recon_loss),
            ]

            self.sum = tf.summary.merge(sums)

            print('... Graph specified.')
            print('---------------------')

    def switch(self, images):
        encodings = [self.encoder(images, 'E_{}'.format(m.name), reuse=True) for m in self.attributes] \
            + [self.encoder(images, 'E_0', reuse=True)]

        # fix identity, switch remaining attributes' values to monitor training process
        transfers = [
            self.decoder(tf.concat([tf.reverse(z, axis=[0]) if i == j else z for i, z in enumerate(encodings)], axis=-1), reuse=True)
            for j in range(1, len(self.attribute_domains))
        ]

        return [images, tf.reverse(images, axis=[0])] + transfers

    def train(self, args, n_epochs, n_decay):
        with tf.Session(graph=self.graph) as sess:
            if not hasattr(self, 'img_size'):
                print('call build_graph first')
                raise
            writer = tf.summary.FileWriter('.logs', self.graph)
            saver = tf.train.Saver()

            if args.from_checkpoint is not None:
                try:
                    print('Trying to load checkpoints...')
                    saver.restore(sess, args.from_checkpoint)
                    print('Checkpoint Loaded.')
                except ValueError:
                    print('Checkpoint "{}" not found.'.format(args.from_checkpoint))
                    exit()
            else:
                print('Initialising variables...')
                sess.run(tf.global_variables_initializer())

            decay = (self.LR / n_decay)

            sess.run(self.train_set.initializer)
            iteration_i = 0
            for epoch_i in range(0, n_epochs + n_decay):
                self.epoch_i = epoch_i
                LR = self.LR - (decay * (epoch_i % n_decay)) if epoch_i > n_epochs else self.LR

                while True:
                    try:
                        feed_dict = {self.LR_PH: LR, self.epoch_PH: epoch_i}

                        # each iteration we randomly permuate the indices
                        for i, m in enumerate(self.attributes):
                            feed_dict[m.idx] = np.random.RandomState(iteration_i + (100 * (i + 10))).permutation(np.array(list(range(self.batch_size))))

                        sess.run([self.optim_G, self.optim_D, self.optim_C, self.X], feed_dict=feed_dict)
                        iteration_i += 1
                    except tf.errors.OutOfRangeError:
                        break

                # reinit training set iterator from the start, and log summary for this epoch
                sess.run(self.train_set.initializer)
                summary, _, _, = sess.run([self.sum, tf.write_file('sample-{}/{}.jpg'.format(args.sample_dir, str(epoch_i).zfill(3)), self.sample), self.X], feed_dict=feed_dict)
                if (epoch_i + 1) % 25 == 0:
                    print('Saving checkpoint...')
                    saver.save(sess, './.checkpoint-{}/model.ckpt'.format(args.sample_dir), global_step=epoch_i + 1)

                print('epoch #{}/{}'.format(epoch_i, n_epochs + n_decay))
                print('LR: {}'.format(LR))
                print('-----------------')

            writer.close()
