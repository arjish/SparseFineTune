import numpy as np
from collections import defaultdict
import os
import argparse
import pickle
from sklearn.metrics import jaccard_score
from scipy.stats.stats import pearsonr

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']
# model_names = ['resnet18', 'resnet34']

data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',
                'traffic_sign', 'quick_draw', 'vgg_flower', 'fungi']

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

parser.add_argument('--nway', default=40, type=int,
    help='number of classes')
parser.add_argument('--kshot', default=1, type=int,
    help='number of shots (support images per class)')

args = parser.parse_args()
nway = args.nway
kshot = args.kshot

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

weights_alldata_pkl = 'weightsEnsembleL1_dict_'+str(nway)+'way.pkl'
if os.path.exists(weights_alldata_pkl):
    with open(weights_alldata_pkl, 'rb') as fp:
        weightsL1_dict = pickle.load(fp)
else:
    print('weights dictionary for all data missing...')

def get_jaccard_among_datasets():
    n_datasets = len(data_folders)
    jaccard_scores = []

    fp = open('jaccard_scores_ensemble_'+str(args.nway)+'way.txt', 'w')
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx1 in range(n_datasets-1):
        for idx2 in range(idx1+1, n_datasets):
            #The order of the backbones have to be fixed!!
            # for model_idx, backbone in enumerate(model_names):
            #     n_top_features = int(features_dim_map[backbone] * 0.2)
            #     top_features_1 = np.argsort(weightsL1_dict[data_folders[idx1]][model_idx])[::-1][:n_top_features]
            #     top_features_2 = np.argsort(weightsL1_dict[data_folders[idx2]][model_idx])[::-1][:n_top_features]
            #     score = jaccard_similarity(top_features_1, top_features_2)
            #     score_null = jaccard_index_null(top_features_1, top_features_2, features_dim_map[backbone])
            #     jaccard_scores[backbone].append((score, score_null))
            #     fp.write(backbone + ": " + data_folders[idx1] + ", " + data_folders[idx2]
            #              + ": " + str(score) + ", " + str(score_null) + "\n")

            top_features_1 = np.argsort(weightsL1_dict[data_folders[idx1]])[::-1][:n_top_features]
            top_features_2 = np.argsort(weightsL1_dict[data_folders[idx2]])[::-1][:n_top_features]
            score = jaccard_similarity(top_features_1, top_features_2)
            jaccard_scores.append(score)
            fp.write( data_folders[idx1] + ", " + data_folders[idx2]
                     + ": " + str(score) + "\n")

    fp.close()
    with open('jaccard_scores_ensemble_'+str(args.nway)+'way.pkl', 'wb') as fp:
        pickle.dump(jaccard_scores, fp)

def get_pearson_coeff_among_datasets():
    n_datasets = len(data_folders)
    pearson_scores = []

    fp = open('pearson_scores_ensemble_' + str(args.nway) + 'way.txt', 'w')
    for idx1 in range(n_datasets - 1):
        for idx2 in range(idx1 + 1, n_datasets):
            features_1 = weightsL1_dict[data_folders[idx1]]
            features_2 = weightsL1_dict[data_folders[idx2]]
            score = [round(s, 4) for s in pearsonr(features_1, features_2)]

            pearson_scores.append(score)
            fp.write(data_folders[idx1] + ", " + data_folders[idx2]
                     + ": " + str(score) + "\n")
    fp.close()

    with open('pearson_scores_ensemble_' + str(args.nway) + 'way.pkl', 'wb') as fp:
        pickle.dump(pearson_scores, fp)

def get_jaccard_fewVsAll():
    weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot) + 'shot' + str(nway) + 'way.pkl'
    if os.path.exists(weights_fewshot_pkl):
        with open(weights_fewshot_pkl, 'rb') as fp:
            weightsL1_fewshot_dict = pickle.load(fp)
    else:
        print('weights dictionary for fewshot missing...')
    n_datasets = len(data_folders)
    jaccard_scores = []

    fp = open('jaccard_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way.txt', 'w')
    n_top_features = int(sum(list(features_dim_map.values())) * 0.2)
    for idx in range(n_datasets):
        top_features_1 = np.argsort(weightsL1_dict[data_folders[idx]])[::-1][:n_top_features]
        top_features_2 = np.argsort(weightsL1_fewshot_dict[data_folders[idx]])[::-1][:n_top_features]
        score = jaccard_similarity(top_features_1, top_features_2)
        jaccard_scores.append(score)
        fp.write(data_folders[idx] + ": " + str(score) + "\n")

    fp.close()
    with open('jaccard_scores_ensemble_fewVsAll_'+str(kshot) + 'shot' + str(args.nway) + 'way.pkl', 'wb') as fp:
        pickle.dump(jaccard_scores, fp)

def get_pearson_coeff_fewVsAll():
    weights_fewshot_pkl = 'weightsEnsembleL1_dict_' + str(kshot) + 'shot' + str(nway) + 'way.pkl'
    if os.path.exists(weights_fewshot_pkl):
        with open(weights_fewshot_pkl, 'rb') as fp:
            weightsL1_fewshot_dict = pickle.load(fp)
    else:
        print('weights dictionary for fewshot missing...')
    n_datasets = len(data_folders)
    pearson_scores = []

    fp = open('pearson_scores_ensemble_fewVsAll_' + str(kshot) + 'shot' + str(args.nway) + 'way.txt', 'w')
    for idx in range(n_datasets):
        features_1 = weightsL1_dict[data_folders[idx]]
        features_2 = weightsL1_fewshot_dict[data_folders[idx]]
        score = [round(s, 4) for s in pearsonr(features_1, features_2)]

        pearson_scores.append(score)
        fp.write(data_folders[idx] + ": " + str(score) + "\n")
    fp.close()

    with open('pearson_scores_ensemble_fewVsAll_' + str(kshot) + 'shot' + str(args.nway) + 'way.pkl', 'wb') as fp:
        pickle.dump(pearson_scores, fp)

def main():
    get_pearson_coeff_among_datasets()
    get_jaccard_among_datasets()
    get_pearson_coeff_fewVsAll()
    get_jaccard_fewVsAll()

if __name__=='__main__':
    main()