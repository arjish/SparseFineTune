import numpy as np
from utils.utils import get_image_features_all
import os, random
import torch
import torch.nn as nn
import argparse
import pickle

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet161', 'densenet169', 'densenet201']
# model_names = ['resnet18', 'resnet34']

data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',  'traffic_sign']
# data_folders = ['birds']

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

count_features = dict()

parser = argparse.ArgumentParser(description='Finetune Classifier')

parser.add_argument('data', help='path to dataset')
parser.add_argument('--nway', default=5, type=int,
    help='number of classes')
parser.add_argument('--kshot', default=1, type=int,
    help='number of shots (support images per class)')
parser.add_argument('--kquery', default=15, type=int,
    help='number of query images per class')
parser.add_argument('--num_epochs', default=200, type=int,
    help='number of epochs')
parser.add_argument('--n_problems', default=50, type=int,
    help='number of test problems')
parser.add_argument('--lr', default=0.001, type=float,
    help='learning rate')
parser.add_argument('--gamma', default=0.8, type=float,
    help='constant value for L2')
parser.add_argument('--nol2', action='store_true', default=False,
    help='set for No L2 regularization, otherwise use L2')

parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()

# Device configuration
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


def train_model(model, features, labels, criterion, optimizer,
                num_epochs=50):
    # Train the model
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    for epoch in range(num_epochs):
        # Move tensors to the configured device
        # x = x.to(device)
        # y = y.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        if not args.nol2:
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


def get_weights(model):
    weights_normed = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights_normed = torch.norm(param, p=1, dim=0).cpu().numpy()

    return weights_normed

def normalize(x):
    tot = sum(x)
    return [round(i/tot, 3) for i in x]

def main():
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    num_epochs = args.num_epochs

    weightsL1_dict = {}
    for dataset in data_folders:
        data_path = os.path.join(args.data, dataset, 'transferred_features_all')
        weightsL1_dict[dataset] = []
        for model_name in model_names:
            meta_folder = os.path.join(data_path, model_name)
            folders = [os.path.join(meta_folder, label) \
                       for label in os.listdir(meta_folder) \
                       if os.path.isdir(os.path.join(meta_folder, label)) \
                       ]

            weights_normed = []
            for i in range(n_problems):
                sampled_folders = random.sample(folders, nway)

                features_support, labels_support, \
                features_query, labels_query = get_image_features_all(sampled_folders,
                                                                      range(nway), nb_samples=n_img, shuffle=True)

                input_size = features_support.shape[1]
                # print('features_query.shape:', features_query.shape)

                model = ClassifierNetwork(input_size, nway).to(device)
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                train_model(model, features_support, labels_support, criterion, optimizer, num_epochs)

                weights_normed.append(get_weights(model))
            weights_normed_mean = np.mean(weights_normed, axis=0)
            weightsL1_dict[dataset].append(weights_normed_mean)

    with open('weightsL1_dict_'+str(args.nway)+'nway.pkl', 'wb') as fp:
        pickle.dump(weightsL1_dict, fp)
        

if __name__=='__main__':
    main()
