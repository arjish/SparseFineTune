import numpy as np
from utils.utils import get_image_features_all
import os, random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse

architecture_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']

parser = argparse.ArgumentParser(description='Finetune Classifier')
parser.add_argument('data', help='path to dataset')
parser.add_argument('--architecture', default='resnet',
    choices=architecture_names, help='model architecture')
parser.add_argument('--domain_type', default='cross',
    choices=['self', 'cross'], help='slef or cross domain testing')
parser.add_argument('--nway', default=5, type=int,
    help='number of classes')
parser.add_argument('--kshot', default=1, type=int,
    help='number of shots (support images per class)')
parser.add_argument('--kquery', default=15, type=int,
    help='number of query images per class')
parser.add_argument('--num_epochs', default=50, type=int,
    help='number of epochs')
parser.add_argument('--n_problems', default=600, type=int,
    help='number of test problems')
parser.add_argument('--hidden_size', default=32, type=int,
    help='hidden layer size')
parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


# Fully connected neural network with one hidden layer
class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassifierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
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
        c = torch.tensor(0.5, device=device)
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
        # x = x.to(device)
        # y = y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted==y).sum().item()

    return 100 * correct / total


def main():
    data = args.data
    architecture = args.architecture
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    domain_type = args.domain_type

    if domain_type=='cross':
        data_path = os.path.join(data, 'transferred_features_test')
    else:
        data_path = os.path.join(data, 'features_test')

    meta_folder = os.path.join(data_path, architecture)

    folders = [os.path.join(meta_folder, label) \
               for label in os.listdir(meta_folder) \
               if os.path.isdir(os.path.join(meta_folder, label)) \
               ]

    accs = []
    for i in range(n_problems):
        sampled_folders = random.sample(folders, nway)

        features_support, labels_support, \
        features_query, labels_query = get_image_features_all(sampled_folders,
            range(nway), nb_samples=n_img, shuffle=True)

        input_size = features_support.shape[1]
        # print('features_query.shape:', features_query.shape)

        model = ClassifierNetwork(input_size, hidden_size, nway).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        train_model(model, features_support, labels_support, criterion, optimizer, num_epochs)

        accuracy_test = test_model(model, features_query, labels_query)

        print(round(accuracy_test, 2))
        accs.append(accuracy_test)

    stds = np.std(accs)
    acc_avg = round(np.mean(accs), 2)
    ci95 = round(1.96 * stds / np.sqrt(n_problems), 2)

    # write the results to a file:
    fp = open('results_finetune.txt', 'a')
    result = 'Setting: ' + domain_type + '-' + data + '- ' + architecture
    result += ': ' + str(nway) + '-way ' + str(kshot) + '-shot'
    result += '; Accuracy: ' + str(acc_avg)
    result += ', ' + str(ci95) + '\n'
    fp.write(result)
    fp.close()

    print("Accuracy:", acc_avg)
    print("CI95:", ci95)


if __name__=='__main__':
    main()
