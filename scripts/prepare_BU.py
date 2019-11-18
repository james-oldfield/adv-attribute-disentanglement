from os import listdir, path, makedirs
from natsort import natsorted
import subprocess

import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='Location of root BU dataset')

args = parser.parse_args()

data_path = args.data_path  #'/home/james/Downloads/BU/'
subjects = natsorted([f for f in listdir(data_path)])

# we randomly choose test set here, as first 10 are relatively uniform, attribute-wise
idx = np.random.RandomState(1234).permutation(list(range(len(subjects))))[:10]

test_ids = list(np.array(subjects)[idx])

emotions = [
    'AN',
    'DI',
    'FE',
    'HA',
    'NE',
    'SA',
    'SU',
]

n_emotions = len(emotions)
n_subjects = len(subjects)


def convert(i, o):
    # shave yellow border 2x2
    return 'convert {} -shave 2x2 +repage -resize 128x128+0+0 {}'.format(i, o)


if not path.exists('./data-bu/'):
    makedirs('./data-bu/')

    for split in ['train', 'test']:
        makedirs('./data-bu/{}'.format(split))

count = 0

for i, s in enumerate(subjects):
    s_path = '{}/{}/'.format(data_path, s)
    for j, file in enumerate([f for f in listdir(s_path) if '.bmp' in f]):
        split = 'test' if s in test_ids else 'train'

        final_path = '{}/{}'.format(s_path, file)

        exp = file.split('_')[1][:2]
        intensity = file.split('_')[1][2:4]

        if count % 100 == 0: print('processed {} images'.format(count))

        exp = emotions.index(exp)

        subprocess.call(convert(final_path, './data-bu/{}/{}-{}-{}-{}.jpg'.format(split, str(i).zfill(4), str(exp).zfill(4), str(intensity).zfill(4), str(count))),
                        shell=True)
        count += 1