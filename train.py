from os import listdir, path, makedirs
from os.path import join
from shutil import copyfile
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import json
from natsort import natsorted

from sklearn.preprocessing import OneHotEncoder

from Model import Model
from Disentangle import Disentangle

##############################
# Define model hyperparameters
##############################
parser = ArgumentParser()
parser.add_argument('--img_size', required=False, type=int, default=128, help='Dimensions of input images (square)')
parser.add_argument('--n_epochs', required=False, type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--n_decay', required=False, type=int, default=25, help='Number of epochs to decay learning rate')
parser.add_argument('--learning_rate', type=float, required=False, default=0.0001, help='ADAM\'s learning rate')
parser.add_argument('--beta1', type=float, required=False, default=0.5, help='Value of Beta1 for ADAM')

parser.add_argument('--from_checkpoint', type=str, required=False, help='Resume from checkpoint')
parser.add_argument('--data_dir', type=str, required=True, help='Location of root image data directory')

parser.add_argument('--n_attributes', type=int, required=True, help='Number of attributes to model')
parser.add_argument('--attribute_names', type=str, required=True, help='Names of the modes of variation to model')

parser.add_argument('--type', type=str, required=True, help='Image Translation (it) or dimensionality reduction (dim)?')

parser.add_argument('--sample_dir', type=str, required=True, help='Location to save samples')

args = parser.parse_args()
args.attribute_names = [str(s) for s in args.attribute_names.split(',')]

# scaffold checkpoint and tensorboard log dirs
if not path.exists('./.checkpoint-{}/'.format(args.sample_dir)):
    makedirs('./.checkpoint-{}/'.format(args.sample_dir))
if not path.exists('./.logs/'):
    makedirs('./.logs/')
if not path.exists('./sample-{}/'.format(args.sample_dir)):
    makedirs('./sample-{}/'.format(args.sample_dir))

copyfile('./Model.py', './sample-{}/Model.py'.format(args.sample_dir))
copyfile('./Disentangle.py', './sample-{}/Disentangle.py'.format(args.sample_dir))
with open('sample-{}/_args.json'.format(args.sample_dir), 'w') as fp:
    json.dump(vars(args), fp)

print('------')
print('Using hyperparams:')
print('------')
print(args)
print('------')

y_train = [[] for _ in range(args.n_attributes)]

train_files = natsorted(listdir(join(args.data_dir, 'train/')))
test_files = natsorted(listdir(join(args.data_dir, 'test/')))

for f in train_files:
    labels = f.split('-')

    if len(labels) < args.n_attributes:
        raise ValueError('Invalid file format (not enough labels for attributes specified)')

    for i in range(args.n_attributes):
        y_train[i].append(int(labels[i]))

train_files = ['{}train/{}'.format(args.data_dir, f) for f in train_files]
test_files = ['{}test/{}'.format(args.data_dir, f) for f in test_files]

labels = [OneHotEncoder().fit_transform(np.expand_dims(y, -1)).toarray().tolist() for y in y_train]

attribute_domains = [len(labels[i][0]) for i, _ in enumerate(range(len(labels)))]

d = Model() if args.type == "it" else Disentangle()
d.build_train_graph(args, train_files, labels, attribute_domains, test_files)
d.train(args, n_epochs=args.n_epochs, n_decay=args.n_decay)
