'''
Note: Use centercrop(299) for inception and centercrop(224) for others in 'val'.
'''

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
#
# model_names = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn',
#                'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#                'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201',
#                'densenet161', 'inception_v3', 'googlenet', 'shufflenet_v2', 'mobilenet_v2',
#                'esnext50_32x4d', 'resnext101_32x8d', 'wideresnet50_2', 'wideresnet101_2', 'mnasnet1_0']

model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'densenet121', 'densenet169', 'densenet201', 'densenet161']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extraction')
parser.add_argument('data', help='path to dataset')
parser.add_argument('-f', '--imageFolderName', default='all')
# parser.add_argument('--model', default='resnet18',
#     choices=model_names, help='model architecture')
parser.add_argument('-n', '--num_classes', default=1000, type=int,
    help='number of classes')
parser.add_argument('--num_epochs', default=15, type=int,
    help='number of epochs')
parser.add_argument('--not_extract', action='store_true', default=False,
    help='set for not extracting, otherwise extract')
parser.add_argument('--train', action='store_true', default=False,
    help='set for train, otherwise validate')
parser.add_argument('-b', '--batch_size', default=32, type=int,
    metavar='N',
    help='mini-batch size (default: 32), this is the total '
         'batch size of all GPUs on the current node when '
         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=0, type=int,
    help='GPU id to use.')

args = parser.parse_args()
# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = not args.not_extract
# model_name = args.model
num_classes = args.num_classes
data_dir = args.data
batch_size = args.batch_size
num_epochs = args.num_epochs

device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
# device = "cpu"

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original_tuple and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def createFolderStructure(model_name):
    imageFolderName = args.imageFolderName
    results_path = os.path.join(args.data, 'transferred_features_'+ imageFolderName, model_name)

    data_path = os.path.join(args.data, imageFolderName)
    classFolders_list = [label \
                         for label in os.listdir(data_path) \
                         if os.path.isdir(os.path.join(data_path, label))]
    for folder_name in classFolders_list:
        if not os.path.exists(os.path.join(results_path, folder_name)):
            os.makedirs(os.path.join(results_path, folder_name))

def train_model(model, dataloaders, criterion, optimizer,
                num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet1_0":
        """ Squeezenet1_0
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
    elif model_name == "densenet201":
        """ Densenet201
        """
        model_ft = models.densenet201(pretrained=use_pretrained)
    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def extract_features(data_loader, model, model_name):
    imageFolderName = args.imageFolderName
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, _, image_path in data_loader:
            # print("Input size:", input.size())
            # compute output
            output_tensor = model(input.to(device))
            output_tensor = nn.AdaptiveAvgPool2d(output_size=(1, 1))(output_tensor)
            # output = output_tensor.detach().numpy()
            output = output_tensor.cpu().numpy()
            output = np.squeeze(output, axis=(2, 3))
            # print("Output shape:", output.shape)
            for i in range(output.shape[0]):
                root, image_name = os.path.split(image_path[i])
                root, folder_name = os.path.split(root)
                save_path = os.path.join(args.data, 'transferred_features_'+imageFolderName, model_name, folder_name)
                # print(save_path)
                np.save(os.path.join(save_path, image_name.split('.')[0]), output[i])

def main():
    for model_name in model_names:
        print("\t Working on model:", model_name)
        # create folders for extracted features
        createFolderStructure(model_name)

        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # Print the model we just instantiated
        # print(model_ft)
        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        print("Initializing Datasets and Dataloaders...")
        image_datasets = {}

        # Create training and validation datasets
        # image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        image_datasets['val'] = ImageFolderWithPaths(os.path.join(data_dir, args.imageFolderName), data_transforms['val'])
        # # Create training and validation dataloaders
        # dataloaders_dict = {
        #     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4) for x in
        #     ['train', 'val']}
        dataloaders_dict = {
            'val': torch.utils.data.DataLoader(image_datasets['val'], shuffle=False, batch_size=batch_size, num_workers=4)}
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        dataset_sizes = {'val': len(image_datasets['val'])}
        # class_names = image_datasets['train'].classes

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.


        # If training:
        if args.train:
            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad==True:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                for name, param in model_ft.named_parameters():
                    if param.requires_grad==True:
                        print("\t", name)
            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                is_inception=(model_name=="inception"))
        else:
            # if args.gpu is None:
            #     model_ft.module = nn.Sequential(*list(model_ft.module.children())[:-1])
            # else:
            #     model_ft = nn.Sequential(*list(model_ft.children())[:-1])

            model_ft = nn.Sequential(*list(model_ft.children())[:-1])
            extract_features(dataloaders_dict['val'], model_ft, model_name)

if __name__ == '__main__':
    main()
