import os
from shutil import copy, copytree
import random
import errno
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split into val, test')
parser.add_argument('data_path', help='path to dataset')
parser.add_argument('--domain_type', default='cross',
    choices=['self', 'cross'], help='self or cross domain testing')

args = parser.parse_args()

data_path = args.data_path
image_path = os.path.join(data_path, 'all')
class_names = [folder \
           for folder in os.listdir(image_path)\
           if os.path.isdir(os.path.join(image_path, folder))]

print('Total number of classes:', len(class_names))

random.seed(1)
random.shuffle(class_names)

percentage_val_class = 50
percentage_test_class = 50
val_test_ratio = [
    percentage_val_class, percentage_test_class]

num_val, num_test = [
    int(float(ratio)/np.sum(val_test_ratio)*len(class_names))
    for ratio in val_test_ratio]

classes = {
    'val': class_names[:num_val],
    'test': class_names[num_val:]
}


if not os.path.exists(os.path.join(data_path, 'classes_val.txt')):
    with open(os.path.join(data_path, 'classes_val.txt'), 'w') as f:
        f.writelines("%s\n" % class_name for class_name in classes['val'])

if not os.path.exists(os.path.join(data_path, 'classes_test.txt')):
    with open(os.path.join(data_path, 'classes_test.txt'), 'w') as f:
        f.writelines("%s\n" % class_name for class_name in classes['test'])


if not os.path.exists(os.path.join(data_path, 'val')):
    os.makedirs(os.path.join(data_path, 'val'))
    for class_name in classes['val']:
        source = os.path.join(image_path, class_name)
        try:
            copytree(source, os.path.join(data_path, 'val', class_name))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno==errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

if not os.path.exists(os.path.join(data_path, 'test')):
    os.makedirs(os.path.join(data_path, 'test'))
    for class_name in classes['test']:
        source = os.path.join(image_path, class_name)
        try:
            copytree(source, os.path.join(data_path, 'test', class_name))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno == errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

