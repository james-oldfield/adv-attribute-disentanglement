from load_model import load_model
import tensorflow as tf
from os import listdir
from os.path import isdir
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--from_checkpoint', type=str, required=True, help='Test from checkpoint')
parser.add_argument('--input_files', type=str, required=True, help='Directory of input source images')
parser.add_argument('--target_files', type=str, required=False, help='Directory of target source images')

parser.add_argument('--n_attributes', type=int, required=True, help='Number of attributes to model')
parser.add_argument('--attribute_names', type=str, required=True, help='Names of the modes of variation to model')
parser.add_argument('--db', type=str, required=True, help='Which database? one of: "bu", "multi", "rafd".')
parser.add_argument('--include_all', action='store_true', help='Include all expressions as input source')

args = parser.parse_args()

if args.target_files is None: args.target_files = args.input_files
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

filter_fns = [lambda _: True, lambda _: True] if args.include_all else filter_fns[args.db]

i_files = ['{}{}'.format(args.input_files, file) for file in list(filter(filter_fns[0], listdir(args.input_files)))] if isdir(args.input_files) else [args.input_files]
t_files = ['{}{}'.format(args.target_files, file) for file in list(filter(filter_fns[1], listdir(args.target_files)))]

batch_size = len(i_files) if len(i_files) < len(t_files) else len(t_files)
if batch_size > 16: batch_size = 16

model = load_model(args.n_attributes, args.attribute_names, batch_size=batch_size, input_files=i_files, target_files=t_files)

with tf.Session(graph=model.graph) as sess:
    x_i = model.X_i
    x_t = model.X_t

    encodings_i = [m.E(x_i) for m in model.attributes] + [model.E_0(x_i)]
    encodings_t = [m.E(x_t, reuse=True) for m in model.attributes]

    transfers = model.decoder(tf.concat([
        encodings_t[i] if i == 1 else z for i, z in enumerate(encodings_i)
    ], axis=-1))

    stack = tf.concat(
        [tf.squeeze(tf.concat(tf.split(img, batch_size), axis=2))
         for img in [x_i, x_t, transfers]], axis=0
    )

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
        sess.run([tf.write_file('transfers/{}-{}.jpg'.format(args.db, str(i).zfill(2)), sample_img), sample_np])
        print('Generatred {}...'.format(i))
        i += 1
    print('... Done')
