from os import listdir, path, makedirs
from natsort import natsorted
import subprocess

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_path', required=True, type=str, help='Location of root MultiPIE dataset')

args = parser.parse_args()

data_path = args.data_path  #'/home/james/Downloads/multiPIE/'
subjects = natsorted([f for f in listdir(data_path)])

# test ids used in the paper
test_ids = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009']

emotions = [
    'neutral',
    'scream',
    'squint',
    'surprise',
]

n_emotions = len(emotions)
n_subjects = len(subjects)


def convert(i, o):
    return 'convert {} -gravity North -crop 220x220+0+0 -resize 128x128+0+0 {}'.format(i, o)


if not path.exists('./data-multi/'):
    makedirs('./data-multi/')

    for split in ['train', 'test']:
        makedirs('./data-multi/{}'.format(split))

count = 0
for i, s in enumerate(subjects):
    for k, e in enumerate(emotions):
        id = str(i).zfill(3)

        split = 'test' if id in test_ids else 'train'
        original_path = '{}{}/{}/{}'.format(data_path, s, '0', e.capitalize())

        for l, light in enumerate(natsorted(listdir(original_path))):
            file_name = listdir('{}/{}'.format(original_path, light))
            f = file_name[0]

            if count % 100 == 0: print('processed {} images'.format(count))

            subprocess.call(convert('{}/{}/{}'.format(original_path, light, f), './data-multi/{}/{}-{}-{}-{}.jpg'.format(split, str(id).zfill(4), str(k).zfill(4), str(l).zfill(4), count)), shell=True)
            count += 1