import json
import shutil
import os
source_path = 'filelists/ILSVRC'

with open(os.path.join(source_path, 'base.json')) as f:
  data_base = json.load(f)
with open(os.path.join(source_path,'val.json')) as f:
  data_val = json.load(f)
with open(os.path.join(source_path,'novel.json')) as f:
  data_novel = json.load(f)

data_labels = data_base['label_names']

print("creating folder structure...")
for i, label in enumerate(data_labels):
    if i %2 == 0:
        dest_folder = os.path.join('train', label)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
    elif i%4 == 1:
        dest_folder = os.path.join('val_oneshot', label)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
    else:
        dest_folder = os.path.join('test', label)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

print("moving training images...")
for img in data_base['image_names']:
    root, file_name = os.path.split(img)
    _, label = os.path.split(root)
    shutil.move(img, os.path.join('train', label))

print("moving val images...")
for img in data_val['image_names']:
    root, file_name = os.path.split(img)
    _, label = os.path.split(root)
    shutil.move(img, os.path.join('val_oneshot', label))

print("moving test images...")
for img in data_novel['image_names']:
    root, file_name = os.path.split(img)
    _, label = os.path.split(root)
    shutil.move(img, os.path.join('test', label))

