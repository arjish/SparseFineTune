import numpy as np
from utils.utils import get_few_features_multiple
import os, random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse

# model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#                'densenet121', 'densenet161', 'densenet169', 'densenet201']

model_names = ['resnet18', 'densenet121']

parser = argparse.ArgumentParser(description='Finetune Classifier')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--domain_type', default='cross',
    choices=['self', 'cross'], help='self or cross domain testing')
parser.add_argument('--nway', default=5, type=int,
    help='number of classes')
parser.add_argument('--kshot', default=1, type=int,
    help='number of shots (support images per class)')
parser.add_argument('--kquery', default=15, type=int,
    help='number of query images per class')
parser.add_argument('--n_problems', default=600, type=int,
    help='number of test problems')
parser.add_argument('--n_trees', default=1000, type=int,
    help='number of trees')
parser.add_argument('--max_depth', default=None, type=int,
    help='max depth of each tree')
parser.add_argument('--max_features', default='auto', choices=['auto', 'sqrt', 'log2'],
    help='max number of features')
parser.add_argument('--max_leaf_nodes', default=None, type=int,
    help='max number of leaf nodes, None means unlimited')
parser.add_argument('--n_jobs', default=-1, type=int,
    help='number of jobs to run in parallel')
parser.add_argument('--ensemble', action='store_true', default=False,
    help='set for random forest, otherwise use one decision tree')


parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()

def main():
    data = args.data
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    domain_type = args.domain_type

    if domain_type=='cross':
        data_path = os.path.join(data, 'transferred_features_all')
    else:
        data_path = os.path.join(data, 'features_test')

    folder_0 = os.path.join(data_path, model_names[0])
    metaval_labels = [label \
                      for label in os.listdir(folder_0) \
                      if os.path.isdir(os.path.join(folder_0, label)) \
                      ]
    labels = metaval_labels
    if args.ensemble:
        clf = RandomForestClassifier(n_estimators=args.n_trees, max_depth=args.max_depth,
            max_features=args.max_features, bootstrap=True,
            max_leaf_nodes=args.max_leaf_nodes,
            n_jobs=args.n_jobs, random_state=0)
    else:
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)

    accs = []
    for i in range(n_problems):
        sampled_labels = random.sample(labels, nway)

        features_support_list, labels_support, \
        features_query_list, labels_query = get_few_features_multiple(data_path, model_names,
                                                                      sampled_labels, range(nway), nb_samples=n_img, shuffle=True)
        features_support = np.concatenate(features_support_list, axis=-1)
        features_query = np.concatenate(features_query_list, axis=-1)
        # print('features_query.shape:', features_query.shape)


        clf.fit(features_support, labels_support)
        predicted = clf.predict(features_query)
        correct = (predicted==labels_query).sum()
        accuracy_test = correct/len(labels_query) * 100

        print(round(accuracy_test, 2))
        accs.append(accuracy_test)

    stds = np.std(accs)
    acc_avg = round(np.mean(accs), 2)
    ci95 = round(1.96 * stds / np.sqrt(n_problems), 2)

    # write the results to a file:
    fp = open('results_finetune.txt', 'a')
    result = 'Setting: Multiple ' + domain_type + '-' + data + '- ' + ', '.join(map(str, model_names))
    if args.ensemble:
        result += '; random_forest, n_trees ' + str(args.n_trees) + ', max_features ' + args.max_features
    else:
        result += ' decision_tree '
    result += ', max_depth ' + str(args.max_depth) + ': ' + str(nway) + '-way ' + str(kshot) + '-shot'
    result += '; Accuracy: ' + str(acc_avg)
    result += ', ' + str(ci95) + '\n'
    fp.write(result)
    fp.close()

    print("Accuracy:", acc_avg)
    print("CI95:", ci95)


if __name__=='__main__':
    main()
