import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import optim, analysis, sampling
#import medmnist
#from medmnist import INFO, Evaluator
from torchvision.models import ResNeXt101_64X4D_Weights
from torchmetrics import AUROC
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader, random_split  
torch.backends.cudnn.benchmark=True
import opacus
from opacus.validators import ModuleValidator
from torchmetrics import AUROC
from statistics import mean
import torch.nn.functional as F


batch_size = 64


pimg_size = (224,224)
mask_size = pimg_size
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
          "epochs":50}






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNISTResNet(nn.Module):
        def __init__(self, in_channels=1):
                super(MNISTResNet, self).__init__()
                # loading a pretrained model
                self.model = torchvision.models.resnet50(pretrained = True)
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
    

def load_datasets():
    # Download and transform CIFAR-10 (train and test)
    # transform = transforms.Compose([
    # transforms.Resize((224,224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean  = (0.49139968, 0.48215827, 0.44653124), std = (0.24703233,
    #                                                                           0.24348505, 0.26158768))])

    transform = transforms.Compose([
     transforms.Resize((224,224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    #train_dataset = datasets.OxfordIIITPet(root='/data', split = 'trainval', download=True, transform=transform)
    #test_dataset = datasets.OxfordIIITPet(root='/data', split = 'test', download=True, transform=transform)

    train_dataset = datasets.CIFAR10(data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform, download=True)


    ######################### NonIID #######################
    # import numpy as np
    # from torch.utils.data import Dataset,Subset
    # labels = np.array(train_dataset.targets)

    # idx = np.where(labels < 3)[0]
    # idx2 = np.where((labels >= 3) & (labels <= 6))[0]
    # idx3 = np.where(labels > 6)[0]


    # train_datasets= []
    # #print(idx)
    # train_datasets.append(Subset(train_dataset,idx))
    # train_datasets.append(Subset(train_dataset,idx2))
    # train_datasets.append(Subset(train_dataset,idx3))
    ############################################################################################

    # train_dataset = datasets.SVHN(data_dir, split = 'train', transform=transform, download=True)
    # test_dataset = datasets.SVHN(data_dir, split = 'test', transform=transform, download=True)


    # train_dataset = DataClass(split='train', transform=transform, download=download)
    # test_dataset  = DataClass(split='test', transform=transform, download=download)

    # Split training set into clients to simulate the individual dataset
    #partition_size = len(train_dataset) // params['total_clients']
    #lengths = [partition_size] * params['total_clients'] ## This line may need more explanation


    #train_datasets = random_split(train_dataset, [len(train_dataset)-2,2])
    #print(len(train_datasets[0]))
    #print(int((len(train_datasets[0])))//3)
    #train_datasets = random_split(train_datasets[0], [int((len(train_dataset))/ params['total_clients']) for _ in range(params['total_clients'])])
    
    #train_datasets = random_split(train_dataset, [int(train_dataset.data.shape[0] / params['total_clients']) for _ in range(params['total_clients'])])

    
    return train_dataset, test_dataset

train_dataset, test_dataset = load_datasets()

def train(model, train_dataset):
  
  
  classifier = model

  # from torch.optim import SGD

  # DPSGD = optim.make_optimizer_class(SGD(momentum=0.9))

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
        params['epochs']
    )

    #iteration = 0
    #acc = []
  for X_minibatch, y_minibatch in minibatch_loader(train_dataset):
        optimizer.zero_grad()
        y_minibatch = torch.squeeze(y_minibatch)
        for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
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


def test(model):

  classifier = model
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
  classifier.eval()
  accuracy = 0.0
  total = 0.0
  a = []
  with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(params['device'])
            labels = torch.squeeze(labels)
            labels = labels.to(params['device'])
            
            # run the model on the test set to predict labels
            outputs = classifier(images)
            #print(outputs.shape[1])
            probs = F.softmax(outputs)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            auroc = AUROC(pos_label=None,num_classes = outputs.shape[1])
            a.append(float(auroc(probs,labels).detach().cpu().numpy()))
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()


  #print(mean(a))
  accuracy = (100 * accuracy / total)
  print('Test Accuracy: {}'.format(accuracy),'                                      AUROC: {}'.format(mean(a)))




def main():

    model = MNISTResNet()
    model = ModuleValidator.fix(model ).to(params['device'])
    ModuleValidator.validate(model , strict=False)
    losses_train = []
    losses_test = []
    acc_train = []
    acc_test = []
    
    for r in range(params["epochs"]):
        loss = 0
        val_acc = 0
        model,_ = train(model, train_dataset)
        # _,val_acc = test(global_model,valloaders[0])
        if(r%1 == 0):
            acc = test(model)
            acc_test.append(acc)
            print('%d-th round' % r)
      #print('average train loss %0.3g  | test acc: %0.3f' % (loss / params["num_sel"], acc))

      #[3.59,16.18,33.85,50.34,60.86,68.25,72.11,74.41,76.15,76.72,77.97]

if __name__ == "__main__":
    main()

