import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import optim, analysis, sampling
import medmnist
from medmnist import INFO, Evaluator
from torchvision.models import ResNeXt101_64X4D_Weights
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader, random_split  
torch.backends.cudnn.benchmark=True

data_flag = 'bloodmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

##### Hyperparameters for federated learning #########
batch_size = 64
num_channels = 3
data_dir = 'data/'

params = {'l2_norm_clip': 1,
          'noise_multiplier' :1.1,
          'minibatch_size': 256,
         'microbatch_size': 1,
         'lr':0.15,
         'l2_penalty' : 0,
         'delta' : 1e-5,
         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
         'local_epochs' : 3,
          "total_clients": 3,
          "num_sel":3,
          "num_rounds": 50}

def load_datasets():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean  = (0.49139968, 0.48215827, 0.44653124), std = (0.24703233,
                                                                              0.24348505, 0.26158768))])

    # transform = transforms.Compose([
    #  transforms.Resize((224,224)),
    # transforms.ToTensor(), 
    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    # train_dataset = datasets.OxfordIIITPet(root='/data', split = 'trainval', download=True, transform=transform)
    # test_dataset = datasets.OxfordIIITPet(root='/data', split = 'test', download=True, transform=transform)

    train_dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)

    # train_dataset = datasets.SVHN(data_dir, split = 'train', transform=transform, download=True)
    # test_dataset = datasets.SVHN(data_dir, split = 'test', transform=transform, download=True)


    #train_dataset = DataClass(split='train', transform=transform, download=download)
    #test_dataset  = DataClass(split='test', transform=transform, download=download)

    # Split training set into clients to simulate the individual dataset
    #partition_size = len(train_dataset) // params['total_clients']
    #lengths = [partition_size] * params['total_clients'] ## This line may need more explanation

    train_datasets = random_split(train_dataset, [len(train_dataset)-2,2])
    train_datasets = random_split(train_datasets[0], [int((len(train_dataset))/ params['total_clients']) for _ in range(params['total_clients'])])
    
    #train_datasets = random_split(train_dataset, [int(train_dataset.data.shape[0] / params['total_clients']) for _ in range(params['total_clients'])])

    
    return train_datasets, test_dataset

train_datasets, test_dataset = load_datasets()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MNISTResNet(nn.Module):
        def __init__(self, in_channels=1):
                super(MNISTResNet, self).__init__()
                # loading a pretrained model
                self.model = torchvision.models.resnet50(pretrained = False)
                # changing the input color channels to 1 since original resnet has 3 channels for RGB
                #self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
                #change the output layer to 10 ckasses as the original resnet has 1000 classes
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, 10)
                # for name,param in self.model.named_parameters():
                #     if(name == 'fc.weight'):
                #         param.requires_grad =True
                #     elif(name =='fc.bias'):
                #         param.requires_grad = True
                #     else:
                #         param.requires_grad = False
        def forward(self, t):
                return self.model(t)

def client_update(client_model, train_dataset):
  
  
  classifier = client_model

  optimizer = optim.DPSGD(
        l2_norm_clip=params['l2_norm_clip'],
        noise_multiplier=params['noise_multiplier'],
        minibatch_size=params['minibatch_size'],
        microbatch_size=params['microbatch_size'],
        params = filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=params['lr'],
        weight_decay=params['l2_penalty'],
    )



    #print('Achieves ({}, {})-DP'.format(analysis.epsilon(len(train_dataset),params['minibatch_size'],params['noise_multiplier'],params['iterations'],params['delta']),params['delta'],))

  loss_function = nn.CrossEntropyLoss()
  minibatch_loader, microbatch_loader = sampling.get_data_loaders(
        params['minibatch_size'],
        params['microbatch_size'],
        params['local_epochs']
    )

    #iteration = 0
    #acc = []
  for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
        optimizer.zero_grad()
        y_minibatch = torch.squeeze(y_minibatch)
        for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
            
            # weights = ResNet50_Weights.DEFAULT
            # preprocess = weights.transforms()
            # X_microbatch = preprocess(X_microbatch)
            X_microbatch = X_microbatch.to(params['device'])
            y_microbatch = y_microbatch.to(params['device'])

            optimizer.zero_microbatch_grad()
            loss = loss_function(classifier(X_microbatch), y_microbatch)
            loss.backward()
            optimizer.microbatch_step()
        optimizer.step()

        # if iteration % 10 == 0:
        #     acc.append(test())
        #     print('Achieves ({}, {})-DP'.format(analysis.epsilon(len(train_dataset),params['minibatch_size'],params['noise_multiplier'],iteration,params['delta']),params['delta'],))
        #     print('[Iteration %d/%d] [Loss: %f]' % (iteration, params['iterations'], loss.item()))

        #iteration += 1

    #return classifier,acc
  return classifier,loss.item() #,val_acc

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """

    global_dict = global_model.state_dict()
    for k in global_dict.keys(): 
      global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
      global_model.load_state_dict(global_dict)
    
    global_model.load_state_dict(global_dict)
    for model in client_models:
      model.load_state_dict(global_model.state_dict())
    
    return client_models

def test(model):

  classifier = model
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
  classifier.eval()
  accuracy = 0.0
  total = 0.0
  with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # weights = ResNet50_Weights.DEFAULT
            # preprocess = weights.transforms()
            # images = preprocess(images)
            images = images.to(params['device'])
            labels = torch.squeeze(labels)
            labels = labels.to(params['device'])
            
            # run the model on the test set to predict labels
            outputs = classifier(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images|
  accuracy = (100 * accuracy / total)
  #print('Test Accuracy: {}'.format(accuracy))
  return accuracy

############################################
#### Initializing models and optimizer  ####
############################################

#### global model ##########
import opacus
from opacus.validators import ModuleValidator
global_model =  MNISTResNet()
global_model = ModuleValidator.fix(global_model ).to(params['device'])
ModuleValidator.validate(global_model , strict=False)

############## client models ##############
client_models = [ ModuleValidator.fix(MNISTResNet()).to(params['device']) for _ in range(params["num_sel"])]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

############### optimizers ################
#opt = [optim.SGD(model.parameters(), lr=0.1) for model in client_models]

#client_models = server_aggregate(global_model, client_models)

parameter_count = 0
for p in global_model.parameters():
  if(p.requires_grad):
    #print(torch.numel(p))
    parameter_count+=torch.numel(p)
    print(p.size())

print(parameter_count)

#net = global_model.to(device)

# for epoch in range(200):
#     global_model,loss= client_update(global_model, train_datasets[0])
#     if(epoch%10==0):
#       acc = test(global_model)
#       print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {acc}")

#  client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]]
#  print(client_idx )

losses_train = []
losses_test = []
acc_train = []
acc_test = []

for r in range(params["num_rounds"]):
    # select random clients
    # "total_clients": 4,
    #       "num_sel":4,
    client_idx = np.random.permutation(params["total_clients"])[:params["num_sel"]]
    #client_idx =[1,2,3]
    # client update
    loss = 0
    temp = 0
    temp1 = 0
    val_acc = 0
    for i in tqdm(range(params["num_sel"])):
      for j in tqdm(range(params['local_epochs'])):
        client_models[i],loss_temp = client_update(client_models[i], train_datasets[client_idx[i]])
      client_models[i].eval()
      loss+=loss_temp 
    losses_train.append(loss)
    losses_train.append(loss)
    # server aggregate
    client_models = server_aggregate(global_model, client_models)

    # _,val_acc = test(global_model,valloaders[0])

    if(r%10==0):
      acc = test(global_model)

      acc_test.append(acc)
      print('%d-th round' % r)
      print('average train loss %0.3g  | test acc: %0.3f' % (loss / params["num_sel"], acc))

test(global_model)

acc_test.append(test(global_model))

acc_test

