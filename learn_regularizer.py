import numpy as np
from utils.utils import get_few_features_multiple, get_all_features_multiple
import os, random
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

data_folders = ['birds', 'aircraft', 'fc100',  'omniglot',  'texture',  'traffic_sign']
# data_folders = ['aircraft']
# model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#                'densenet121', 'densenet161', 'densenet169', 'densenet201']
model_names = ['resnet18', 'densenet121']

parser = argparse.ArgumentParser(description='Finetune Classifier')
parser.add_argument('data', help='path to dataset')
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
parser.add_argument('--lr', default=0.001, type=float,
    help='learning rate')
parser.add_argument('--l2', action='store_true', default=False,
    help='set for L2 regularization, otherwise no regularization')
parser.add_argument('--gamma', default=0.5, type=float,
    help='constant value for L2')
parser.add_argument('--linear', action='store_true', default=False,
    help='set for linear model, otherwise use hidden layer')

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


def train_model(model, trainloader, features_fewshot, labels_fewshot,
                criterion, optimizer, optimizer_regularizer, r_tensor, epoch, dataset):
    # Train the model
    x_fewshot = torch.tensor(features_fewshot, dtype=torch.float32, device=device)
    y_fewshot = torch.tensor(labels_fewshot, dtype=torch.long, device=device)
    # Train the model
    model.train()  # Set model to training mode
    param_list = list(model.parameters())

    # Move tensors to the configured device
    for x, y in trainloader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(x)
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

        # model_fewshot = deepcopy(model)
        output_fewshot = model(x_fewshot)  # copy the model parameters and not use the same ones
        params = torch.cat([torch.flatten(param) for param in param_list], dim =0)
        grads_big = torch.cat([torch.flatten(param.grad) for param in param_list], dim=0)
        # grads_big = [param.grad for param in param_list]
        loss_fewshot = criterion(output_fewshot, y_fewshot) + torch.dot(r_tensor, torch.square(params))
        loss_fewshot.backward(create_graph=True)
        grads_small = torch.cat([torch.flatten(param.grad) for param in list(model.parameters())], dim=0)
        # grads_small = [param.grad for param in list(model.parameters())]

        torch.autograd.set_detect_anomaly(True)
        output_regularizer = grads_small
        target_regularizer = grads_big
        loss_regularizer = torch.nn.MSELoss()(output_regularizer, target_regularizer)
    if epoch % 10 == 0:
        # print("Epoch:", epoch, "Classification loss", loss.data.cpu().numpy().item())
        print("\t\t", dataset, "Epoch:", epoch, "\t\tLoss", loss_regularizer.data.cpu().numpy().item())

    optimizer_regularizer.zero_grad()
    loss_regularizer.backward()
    optimizer_regularizer.step()

    return r_tensor




def main():
    nway = args.nway
    kshot = args.kshot
    kquery = args.kquery
    n_img = kshot + kquery
    n_problems = args.n_problems
    num_epochs = args.num_epochs

    models = []
    trainloaders = []
    optimizers = []
    for idx, dataset in enumerate(data_folders):
        data_path = os.path.join(args.data, dataset, 'transferred_features_all')
        folder_0 = os.path.join(data_path, model_names[0])
        label_folders = [label \
                         for label in os.listdir(folder_0) \
                         if os.path.isdir(os.path.join(folder_0, label)) \
                         ]
        sampled_label_folders = random.sample(label_folders, nway)

        features, label_folders = get_all_features_multiple(data_path, model_names,
                                                            sampled_label_folders, range(nway))
        input_size = features.shape[1]

        train_data = []
        for i in range(features.shape[0]):
            train_data.append([features[i], label_folders[i]])

        models.append(ClassifierNetwork(input_size, nway).to(device))
        trainloaders.append(torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=200))

        optimizers.append(torch.optim.Adam(models[idx].parameters(), lr=args.lr))

    # to get the model parameter size:
    num_params = sum([param.numel() for param in list(models[0].parameters())])

    if os.path.exists("regularizer_weights.npy"):
        r = np.load("regularizer_weights.npy")
    else:
        r = np.random.normal(0, 0.1, num_params).astype(np.float32)
    if device.type == "cpu":
        # dtype = torch.FloatTensor
        r_tensor = Variable(torch.from_numpy(r), requires_grad=True)
    else:
        # dtype = torch.cuda.FloatTensor
        r_tensor = Variable(torch.from_numpy(r).cuda(), requires_grad=True)
    optimizer_regularizer = torch.optim.Adam([r_tensor], lr=0.05)

    # Full data training:
    criterion = nn.CrossEntropyLoss()

    for prob in range(n_problems):
        print('Problem num:', prob)
        for epoch in range(num_epochs):
            for idx, dataset in enumerate(data_folders):
                if epoch == 0:
                    data_path = os.path.join(args.data, dataset, 'transferred_features_all')
                    folder_0 = os.path.join(data_path, model_names[0])
                    label_folders = [label \
                                     for label in os.listdir(folder_0) \
                                     if os.path.isdir(os.path.join(folder_0, label)) \
                                     ]
                    sampled_label_folders = random.sample(label_folders, nway)

                    features_support_list, labels_support, \
                            features_query_list, labels_query = get_few_features_multiple(kshot, data_path, model_names,
                                                                                          sampled_label_folders, range(nway), nb_samples=n_img, shuffle=True)
                    features_support = np.concatenate(features_support_list, axis=-1)

                # train one epoch
                r_tensor = train_model(models[idx], trainloaders[idx], features_support, labels_support,
                                criterion, optimizers[idx], optimizer_regularizer, r_tensor, epoch, dataset)

                if prob % 10 == 0:
                    np.save('regularizer_weights', r_tensor.cpu().detach().numpy())

if __name__=='__main__':
    main()
