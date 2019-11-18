from load_model import load_model
import tensorflow as tf
from os import listdir
from os.path import isdir
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--from_checkpoint', type=str, required=True, help='Test from checkpoint')
parser.add_argument('--input_files', type=str, required=True, help='Directory of input source images')

parser.add_argument('--n_attributes', type=int, required=True, help='Number of attributes to model')
parser.add_argument('--attribute_names', type=str, required=True, help='Names of the modes of variation to model')
parser.add_argument('--db', type=str, required=True, help='Which database? one of: "bu", "multi", "rafd".')

args = parser.parse_args()

args.target_files = args.input_files
args.attribute_names = [str(s) for s in args.attribute_names.split(',')]

# filter the input and target files respectively if desired (e.g. select for certain lighting conditions for model comparison)
filter_fns = {
    'bu': [
        (lambda x: int(x.split('-')[1]) == 4),  # neutral expression input
        (lambda x: int(x.split('-')[2]) == 4),  # intensity 4
    ],
    'multi': [
        (lambda x: int(x.split('-')[1]) == 0),  # neutral expression input
        (lambda x: True),
    ],
    'rafd': [
        (lambda x: int(x.split('-')[1]) == 5),  # neutral expression input
        (lambda x: True),
    ],
}

filter_fns = filter_fns[args.db]

i_files = ['{}{}'.format(args.input_files, file) for file in list(filter(filter_fns[0], listdir(args.input_files)))] if isdir(args.input_files) else [args.input_files]
t_files = ['{}{}'.format(args.target_files, file) for file in list(filter(filter_fns[1], listdir(args.target_files)))]

batch_size = len(i_files) if len(i_files) < len(t_files) else len(t_files)
if batch_size > 16: batch_size = 16

model = load_model(args.n_attributes, args.attribute_names, batch_size=batch_size, input_files=i_files, target_files=t_files)

with tf.Session(graph=model.graph) as sess:
    x_i = model.X_i[:1]
    x_t = model.X_t[:2]

    encodings_i = [m.E(x_i) for m in model.attributes] + [model.E_0(x_i)]
    encodings_t = [m.E(x_t, reuse=True) for m in model.attributes]

    transfers = []

    for i in range(10):
        s = i / 10.
        transfers.append(
            tf.squeeze(model.decoder(tf.concat([
                encodings_i[0],
                tf.expand_dims((1. - s) * encodings_t[1][0] + s * encodings_t[1][1], 0),
                encodings_i[2],
            ], axis=-1), reuse=tf.AUTO_REUSE)))

    stack = tf.squeeze(tf.concat(tf.split([x_t[0]] + transfers + [x_t[1]], 12), axis=2))

    sample_np = tf.clip_by_value(tf.divide(tf.add(stack, 1.0), 2.0), 0, 255)
    sample_img = tf.image.convert_image_dtype(sample_np, saturate=True, dtype=tf.uint16)
    sample_img = tf.image.encode_png(sample_img)

    saver = tf.train.Saver()

    try:
        print('Trying to load checkpoints...')
        saver.restore(sess, args.from_checkpoint)
        print('Checkpoint Loaded.')
    except ValueError:
        print('Checkpoints not loaded / found.')
        raise

    i = 0
    for _ in range(100):
        sess.run([tf.write_file('joint-transfers/{}-{}.jpg'.format(args.db, str(i).zfill(2)), sample_img), sample_np])
        print('Generatred {}...'.format(i))
        i += 1
    print('... Done')
