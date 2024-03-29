import numpy as np
from utils.utils import get_few_features_multiple
import os, random
import torch
import torch.nn as nn
from math import ceil
import torchvision
import torchvision.transforms as transforms
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
parser.add_argument('--num_epochs', default=100, type=int,
    help='number of epochs')
parser.add_argument('--n_problems', default=600, type=int,
    help='number of test problems')
parser.add_argument('--hidden_size1', default=1024, type=int,
    help='hidden layer size')
parser.add_argument('--hidden_size2', default=128, type=int,
    help='hidden layer size')
parser.add_argument('--lr', default=0.001, type=float,
    help='learning rate')
parser.add_argument('--l2', action='store_true', default=False,
    help='set for L2 regularization, otherwise no regularization')
parser.add_argument('--gamma', default=0.5, type=float,
    help='constant value for L2')
parser.add_argument('--linear', action='store_true', default=False,
    help='set for linear model, otherwise use hidden layer')
parser.add_argument('--reg_file', default='regularizer_weights_cosine.npy',
    help='self or cross domain testing')

parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()

# Device configuration
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierNetwork, self).__init__()
        if not args.linear:
            self.fc1 = nn.Linear(input_size, args.hidden_size1)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(args.hidden_size1, num_classes)
        else:
            self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        if not args.linear:
            out = self.tanh(out)
            out = self.fc2(out)
        return out


def train_model(model, features, labels, criterion, optimizer,
                r, num_epochs):
    # Train the model
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        params = torch.cat([torch.flatten(param) for param in list(model.parameters())], dim=0)
        if r is not None:
            loss = criterion(outputs, y) + torch.dot(torch.square(torch.tensor(r, device=device)), params)
        else:
            loss = criterion(outputs, y)
        if args.l2:
            c = torch.tensor(args.gamma, device=device)
            l2_reg = torch.tensor(0., device=device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg += torch.norm(param)

            loss += c * l2_reg

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('Epoch [{}/{}],  Loss: {:.4f}'
        #     .format(epoch + 1, num_epochs, loss.item()))


def test_model(model, features, labels):
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        correct = 0
        total = 0
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted==y).sum().item()

    return 100 * correct / total


def main():
    data = args.data
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    num_epochs = args.num_epochs
    domain_type = args.domain_type

    if domain_type=='cross':
        data_path = os.path.join(data, 'transferred_features_all')
    else:
        data_path = os.path.join(data, 'transferred_features_test')

    folder_0 = os.path.join(data_path, model_names[0])
    labels = [label \
                      for label in os.listdir(folder_0) \
                      if os.path.isdir(os.path.join(folder_0, label)) \
                      ]

    if os.path.exists(args.reg_file):
        print("Using regularizer from", args.reg_file)
        r = np.load(args.reg_file)
    else:
        r = None
    if args.l2:
        print("Using L2 regularizer")

    accs = []
    for i in range(n_problems):

        sampled_labels = random.sample(labels, nway)

        features_support_list, labels_support, \
        features_query_list, labels_query = get_few_features_multiple(kshot, data_path, model_names,
                                          sampled_labels, range(nway), nb_samples=n_img, shuffle=True)
        features_support = np.concatenate(features_support_list, axis=-1)
        features_query = np.concatenate(features_query_list, axis=-1)

        input_size = features_support.shape[1]
        # print('features_query.shape:', features_query.shape)

        model = ClassifierNetwork(input_size, nway).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_model(model, features_support, labels_support, criterion, optimizer, r, num_epochs)

        accuracy_test = test_model(model, features_query, labels_query)

        print(round(accuracy_test, 2))
        accs.append(accuracy_test)

    stds = np.std(accs)
    acc_avg = round(np.mean(accs), 2)
    ci95 = round(1.96 * stds / np.sqrt(n_problems), 2)

    # write the results to a file:
    fp = open('results_finetune.txt', 'a')
    result = 'Setting: Multiple ' + domain_type + '-' + data + '- ' + ', '.join(map(str, model_names))
    if args.linear:
        result += ' linear'
    if args.l2:
        result += ' L2'
    result += ': ' + str(nway) + '-way ' + str(kshot) + '-shot'
    result += '; Accuracy: ' + str(acc_avg)
    result += ', ' + str(ci95) + '\n'
    fp.write(result)
    fp.close()

    print("Accuracy:", acc_avg)
    print("CI95:", ci95)


if __name__=='__main__':
    main()
