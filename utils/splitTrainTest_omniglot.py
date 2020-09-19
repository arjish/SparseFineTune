import os
from shutil import copy, copytree
import random
import errno

data_path = 'omniglot-master/omniglot'


character_folders = [os.path.join(data_path, family, character) \
                for family in os.listdir(data_path) \
                if os.path.isdir(os.path.join(data_path, family)) \
                for character in os.listdir(os.path.join(data_path, family))]

print('Total number of character folders:', len(character_folders))

random.seed(1)
random.shuffle(character_folders)

num_train = 1200

train_folders = character_folders[:num_train]
test_folders = character_folders[num_train:]


if not os.path.exists(os.path.join(data_path, 'omniglot_folders_train.txt')):
    with open(os.path.join(data_path, 'omniglot_folders_train.txt'), 'w') as f:
        f.writelines("%s\n" % folder for folder in train_folders)

if not os.path.exists(os.path.join(data_path, 'omniglot_folders_test.txt')):
    with open(os.path.join(data_path, 'omniglot_folders_test.txt'), 'w') as f:
        f.writelines("%s\n" % folder for folder in test_folders)


if not os.path.exists(os.path.join(data_path, 'train')):
    os.makedirs(os.path.join(data_path, 'train'))
    for folder in train_folders:
        root, char_folder = os.path.split(folder)
        root, alphabet_folder = os.path.split(root)
        try:
            copytree(folder, os.path.join(root, 'train', alphabet_folder, char_folder))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno==errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

if not os.path.exists(os.path.join(data_path, 'test')):
    os.makedirs(os.path.join(data_path, 'test'))
    for folder in test_folders:
        root, char_folder = os.path.split(folder)
        root, alphabet_folder = os.path.split(root)
        try:
            copytree(folder, os.path.join(root, 'test', alphabet_folder, char_folder))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno == errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

