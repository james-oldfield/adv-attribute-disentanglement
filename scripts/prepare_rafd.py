from os import listdir, path, makedirs
from natsort import natsorted
import subprocess

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='Location of root RaFD dataset')

args = parser.parse_args()

data_path = args.data_path  #'/home/james/Downloads/multiPIE/'
subjects = natsorted([f for f in listdir(data_path)])

# test ids used in the paper
test_ids = ['01', '02', '03', '04', '05', '07', '08', '64']

emotions = [
    'angry',
    'contemptuous',
    'disgusted',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised',
]

glances = [
    'frontal',
    'left',
    'right',
]

n_emotions = len(emotions)
n_subjects = len(subjects)


def convert(i, o):
    return 'convert {} -gravity Center -crop 640x640+0-128 +repage -resize 128x128+0+0 {}'.format(i, o)


def flip(i, o):
    return 'convert {} -gravity Center -crop 640x640+0-128 +repage -resize 128x128+0+0 -flop {}'.format(i, o)


if not path.exists('./data-rafd/'):
    makedirs('./data-rafd/')

    for split in ['train', 'test']:
        makedirs('./data-rafd/{}'.format(split))

count = 0
for i, s in enumerate(subjects):
    parts = s.split('_')
    emotion = emotions.index(parts[-2])
    id = parts[1]
    glance = glances.index(parts[-1].split('.')[-2])

    split = 'test' if id in test_ids else 'train'

    if '090' not in s: continue

    if count % 100 == 0: print('done, ', count)

    subprocess.call(convert('{}{}'.format(data_path, s), './data-rafd/{}/{}-{}-{}-{}.jpg'.format(split, str(id).zfill(4), str(emotion).zfill(4), str(glance).zfill(4), str(count).zfill(4))), shell=True)
    count += 1

    # duplicate data
    if split == 'train':
        # switch glance labels for flip
        if glance == 1: glance = '2'
        if glance == 2: glance = '1'
        subprocess.call(flip('{}{}'.format(data_path, s), './data-rafd/{}/{}-{}-{}-{}.jpg'.format(split, str(id).zfill(4), str(emotion).zfill(4), str(glance).zfill(4), str(count).zfill(4))), shell=True)
        count += 1