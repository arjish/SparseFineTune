import numpy as np
from collections import defaultdict
import argparse
import pickle
from sklearn.metrics import jaccard_score

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']
# model_names = ['resnet18', 'resnet34']

data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',  'traffic_sign']

features_dim_map = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920
}

parser = argparse.ArgumentParser(description='Get Jaccard Index')

parser.add_argument('--nway', default=5, type=int,
    help='number of classes')

args = parser.parse_args()

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return round(len(s1.intersection(s2)) / len(s1.union(s2)), 3)

def jaccard_index_null (list1, list2, n_features):
    n1 = len(list1)
    n2 = len(list2)
    # assert len(s1) == len(s2), "Lengths of two sets are not same"

    term = (n1 * n2)/n_features
    return round(term / (n1 + n2 - term), 3)

with open('weightsL1_dict_' + str(args.nway) + 'nway.pkl', 'rb') as fp:
    weightsL1_dict = pickle.load(fp)

n_datasets = len(data_folders)
jaccard_scores = defaultdict(list)

fp = open('jaccard_scores'+str(args.nway)+'way.txt', 'w')
for idx1 in range(n_datasets-1):
    for idx2 in range(idx1+1, n_datasets):
        #The order of the backbones have to be fixed!!
        for model_idx, backbone in enumerate(model_names):
            n_top_features = int(features_dim_map[backbone] * 0.2)
            top_features_1 = np.argsort(weightsL1_dict[data_folders[idx1]][model_idx])[::-1][:n_top_features]
            top_features_2 = np.argsort(weightsL1_dict[data_folders[idx2]][model_idx])[::-1][:n_top_features]
            score = jaccard_similarity(top_features_1, top_features_2)
            score_null = jaccard_index_null(top_features_1, top_features_2, features_dim_map[backbone])
            jaccard_scores[backbone].append((score, score_null))
            fp.write(backbone + ": " + data_folders[idx1] + ", " + data_folders[idx2]
                     + ": " + str(score) + ", " + str(score_null) + "\n")
fp.close()
with open('jaccard_scores'+str(args.nway)+'way.pkl', 'wb') as fp:
    pickle.dump(jaccard_scores, fp)
