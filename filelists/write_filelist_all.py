import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import argparse
import random

parser = argparse.ArgumentParser(description='Write filelist')
parser.add_argument('dataset', help='name of the dataset')
parser.add_argument('--datapath', default="../../../data/",
                    help='parent path to the data')
args = parser.parse_args()

data_path = os.path.abspath(args.datapath)
test_path = join(data_path, args.dataset, 'all')

savedir = './'
dataset_list = ['novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list_test = [f for f in listdir(test_path) if isdir(join(test_path, f))]

folder_list = [folder for folder in folder_list_test]
# label_dict_train = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_train, classfile_list_val, classfile_list_test = [], [], []


for i, folder in enumerate(folder_list_test):
    folder_path = join(test_path, folder)
    classfile_list_test.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_test[i])


label_count = 0
for dataset in dataset_list:
    file_list = []
    label_list = []
    classfile_list = []
    if 'base' in dataset:
        classfile_list = classfile_list_train
    elif 'val' in dataset:
        classfile_list = classfile_list_val
    else:
        classfile_list = classfile_list_test

    for i, classfile_list in enumerate(classfile_list):
        file_list = file_list + classfile_list
        label_list = label_list + np.repeat(label_count, len(classfile_list)).tolist()
        label_count += 1

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
