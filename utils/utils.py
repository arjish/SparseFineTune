from random import sample, shuffle
import random
import numpy as np
import os
from copy import deepcopy


def get_files(path, nb_samples=None, shuffle=True, multi_path=False):
    if nb_samples is not None:
        sampler = lambda x: np.random.choice(x, nb_samples)
    else:
        sampler = lambda x: x

    if multi_path:
        files = [os.path.join(item, image) \
                 for item in path \
                 for image in sampler(os.listdir(item))]
    else:
        files = [os.path.join(path, image) \
                 for image in sampler(os.listdir(path))]

    if shuffle:
        random.shuffle(files)
    return files  # list of files


def get_image_features(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    files_labels = [(i, os.path.join(path, file)) \
              for i, path in zip(labels, paths) \
              for file in sampler(os.listdir(path))]

    if shuffle:
        random.shuffle(files_labels)

    labels = [fl[0] for fl in files_labels]
    files = [fl[1] for fl in files_labels]

    features = np.array([np.load(file) for file in files])

    # Add a dummy feature of value '1' (at 0th index) for all the images ::
    features = np.hstack((np.ones((len(files), 1)), features))

    return features, labels

def get_image_features_all(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    files_labels = [(i, os.path.join(path, file)) \
              for i, path in zip(labels, paths) \
              for file in sampler(os.listdir(path))]

    nway = len(paths)
    support_ids = [i for i in range(nb_samples*nway) if i%nb_samples==0]
    query_ids = [i for i in range(nb_samples * nway) if i % nb_samples!=0]
    files_labels_support = [files_labels[i] for i in support_ids]
    files_labels_query = [files_labels[i] for i in query_ids]

    if shuffle:
        random.shuffle(files_labels_support)
        random.shuffle(files_labels_query)

    labels_support = [fl[0] for fl in files_labels_support]
    files_support = [fl[1] for fl in files_labels_support]
    labels_query = [fl[0] for fl in files_labels_query]
    files_query = [fl[1] for fl in files_labels_query]

    features_support = np.array([np.load(file) for file in files_support])
    features_query = np.array([np.load(file) for file in files_query])

    return features_support, labels_support, features_query, labels_query

def get_image_features_clusters(meta_folder, sampled_labels,
            n_clusters, labels, nb_samples, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    cluster_paths_0 = [os.path.join(meta_folder,
        'cluster_'+str(n_clusters)+'_0', item) for item in sampled_labels]
    sampled_files_labels = [(i, os.path.join(os.path.split(p)[-1], file)) \
              for i, p in zip(labels, cluster_paths_0) \
              for file in sampler(os.listdir(p))]

    nway = len(sampled_labels)
    support_ids = [i for i in range(nb_samples*nway) if i%nb_samples==0]
    query_ids = [i for i in range(nb_samples * nway) if i % nb_samples!=0]
    files_labels_support = [sampled_files_labels[i] for i in support_ids]
    files_labels_query = [sampled_files_labels[i] for i in query_ids]

    if shuffle:
        random.shuffle(files_labels_support)
        random.shuffle(files_labels_query)

    labels_support = [fl[0] for fl in files_labels_support]
    files_support = [fl[1] for fl in files_labels_support]
    labels_query = [fl[0] for fl in files_labels_query]
    files_query = [fl[1] for fl in files_labels_query]

    features_support_list = []
    features_query_list = []
    for cl in range(n_clusters):
        cluster_path = os.path.join(meta_folder,
            'cluster_' + str(n_clusters) + '_' + str(cl))

        features_support_list.append(np.array([np.load(os.path.join(cluster_path, file))
                                                for file in files_support]))
        features_query_list.append(np.array([np.load(os.path.join(cluster_path, file))
                                                for file in files_query]))

    features_support = np.concatenate(features_support_list, axis=-1)
    features_query = np.concatenate(features_query_list, axis=-1)

    return features_support, labels_support, features_query, labels_query


def get_image_features_multiple(data_path, model_folders,
            sampled_labels, labels, nb_samples, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    folder_0 = [os.path.join(data_path, model_folders[0], item) for item in sampled_labels]
    sampled_files_labels = [(i, os.path.join(os.path.split(p)[-1], file)) \
              for i, p in zip(labels, folder_0) \
              for file in sampler(os.listdir(p))]

    nway = len(sampled_labels)
    support_ids = [i for i in range(nb_samples*nway) if i%nb_samples==0]
    query_ids = [i for i in range(nb_samples * nway) if i % nb_samples!=0]
    files_labels_support = [sampled_files_labels[i] for i in support_ids]
    files_labels_query = [sampled_files_labels[i] for i in query_ids]

    if shuffle:
        random.shuffle(files_labels_support)
        random.shuffle(files_labels_query)

    labels_support = [fl[0] for fl in files_labels_support]
    files_support = [fl[1] for fl in files_labels_support]
    labels_query = [fl[0] for fl in files_labels_query]
    files_query = [fl[1] for fl in files_labels_query]

    features_support_list = []
    features_query_list = []
    for model in model_folders:
        model_path = os.path.join(data_path, model)
        features_support_list.append(np.array([np.load(os.path.join(model_path, file))
                                                for file in files_support]))
        features_query_list.append(np.array([np.load(os.path.join(model_path, file))
                                                for file in files_query]))

    features_support = np.concatenate(features_support_list, axis=-1)
    features_query = np.concatenate(features_query_list, axis=-1)

    return features_support, labels_support, features_query, labels_query