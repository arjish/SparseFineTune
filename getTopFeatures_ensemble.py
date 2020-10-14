import numpy as np
from utils.utils import get_image_features_multiple
import os, random
import torch
import torch.nn as nn
import argparse
from collections import Counter

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

    weightsL1_data = []
    top_features = []
    for dataset in data_folders:
        data_path = os.path.join(args.data, dataset, 'transferred_features_all')

        folder_0 = os.path.join(data_path, model_names[0])
        metaval_labels = [label \
                          for label in os.listdir(folder_0) \
                          if os.path.isdir(os.path.join(folder_0, label)) \
                          ]
        labels = metaval_labels

        weights_normed = []
        for i in range(n_problems):
            sampled_labels = random.sample(labels, nway)

            features_support_list, labels_support, \
            features_query_list, labels_query = get_image_features_multiple(data_path, model_names,
                                                                            sampled_labels, range(nway), nb_samples=n_img,
                                                                            shuffle=True)
            features_support = np.concatenate(features_support_list, axis=-1)
            # features_query = np.concatenate(features_query_list, axis=-1)

            input_size = features_support.shape[1]
            # print('features_query.shape:', features_query.shape)

            model = ClassifierNetwork(input_size, nway).to(device)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            train_model(model, features_support, labels_support, criterion, optimizer, num_epochs)

            weights_normed.append(get_weights(model))
        weights_normed_mean = np.mean(weights_normed, axis=0)
        n_top_features = int(input_size * 0.1)

        # print('weights_normed_mean shape:', weights_normed_mean.shape)

        avg_per_model = []
        start_idx = 0
        backbone_weight_list = []
        for idx, model_name in enumerate(model_names):
            length = features_dim_map[model_name]
            current_weights = weights_normed_mean[start_idx: start_idx+length]
            # avg_per_model.append(np.mean(current_weights))
            backbone_weight_list.extend([(idx, item) for item in current_weights])
            start_idx = start_idx+length
        # weightsL1_data.append(avg_per_model)

        top_backbones = sorted(backbone_weight_list, key=lambda x: x[1], reverse=True)[:n_top_features]
        top_backbones = [item[0] for item in top_backbones]
        top_features.append(Counter(top_backbones))

    # weightsL1_data = [normalize(i) for i in weightsL1_data]
    # print(weightsL1_data)
    print(top_features)
    # np.save('weightsL1_data', weightsL1_data)
    np.save('top_features'+str(nway)+'way', top_features)



        

if __name__=='__main__':
    main()
