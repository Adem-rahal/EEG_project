from torch.optim import AdamW
from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Set, Tuple
from sklearn.metrics import confusion_matrix

import seaborn as sn
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T 
import sys
import os
import matplotlib.pyplot as plt


cuda = torch.device('cuda:1')
###################### DATASET & MODEL INITIALIZATION ##################################

# class EEGData(Dataset) :
#     def __init__(self,dir: str="Training", lenght: int=0, transform=None) -> None:
#         self.dir = dir
#         self.lenght = lenght

#     def __len__(self) -> int:
#         return self.lenght

#     def __getitem__(self, idx) -> Tuple[torch.Tensor,int]:
#         return pick_data(self.dir,idx)

class EEGData(Dataset) :
    def __init__(self, x, y, transform=None) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor,int]:
        return (torch.from_numpy(self.x[idx]).float(),int(self.y[idx]))

# class Model(nn.Module):     #BEST : 49,34% LR=10e-5
#     def __init__(self, channels: int=8, n_classes: int=3)->None:
#         super(Model,self).__init__()
#         self.features = nn.Sequential(
#             nn.BatchNorm1d(channels),nn.Conv1d(channels,64,kernel_size=3,stride=1),nn.ReLU(),nn.BatchNorm1d(64),nn.Dropout(0.4),
#             nn.Conv1d(64,128,kernel_size=2,stride=2),nn.ReLU(),nn.BatchNorm1d(128),nn.MaxPool1d(2),
#             nn.Conv1d(128,256,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
#             nn.Flatten(),
#             #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(768),nn.Linear(768,512),nn.ReLU(),nn.Dropout(0.4),
#             nn.Linear(512,n_classes),
#         )

#     def forward (self, x: torch.Tensor)->torch.Tensor:
#         # print(x.size())
#         x=self.features(x)
#         # print(x.size())
#         x=self.classifier(x)
#         # print(x.size())
#         return x


class Model(nn.Module):     
    def __init__(self, channels: int=8, n_classes: int=3)->None:
        super(Model,self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels,64,kernel_size=3,stride=1),nn.ReLU(),nn.Dropout(0.4),
            nn.Conv1d(64,128,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,256,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Flatten(),
            #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,n_classes),
        )

    def forward (self, x: torch.Tensor)->torch.Tensor:
        # print(x.size())
        x=self.features(x)
        # print(x.size())
        x=self.classifier(x)
        # print(x.size())
        return x

class Model(nn.Module):     
    def __init__(self, channels: int=8, n_classes: int=3)->None:
        super(Model,self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels,32,kernel_size=2),nn.ReLU(),nn.BatchNorm1d(32),nn.MaxPool1d(2),
            nn.Conv1d(32,32,kernel_size=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.5),
            nn.Conv1d(32,64,kernel_size=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,64,kernel_size=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.5),
            nn.Conv1d(64,128,kernel_size=1),nn.ReLU(),nn.MaxPool1d(2),
            # nn.Conv1d(128,128,kernel_size=1,stride=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.5),
            # nn.Conv1d(128,256,kernel_size=1),nn.ReLU(),nn.MaxPool1d(2),
            # nn.Conv1d(256,256,kernel_size=1),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.5),
            # nn.Conv1d(256,512,kernel_size=1),nn.ReLU(),nn.MaxPool1d(2),
            # nn.Conv1d(512,512,kernel_size=1),nn.ReLU(),nn.MaxPool1d(2),
            nn.Flatten(),
            #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(128,32),
            nn.Linear(32,n_classes),
        )

    def forward (self, x: torch.Tensor)->torch.Tensor:
        # print(x.size())
        x=self.features(x)
        # print(x.size())
        x=self.classifier(x)
        # print(x.size())
        return x


# class Model(nn.Module):
#     def __init__(self, channels: int=8, n_classes: int=3)->None:
#         super(Model,self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv1d(channels,64,kernel_size=3),nn.ReLU(),nn.Dropout(0.4),
#             nn.Conv1d(64,64,kernel_size=2),nn.ReLU(),nn.MaxPool1d(2),
#             nn.Conv1d(64,64,kernel_size=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
#             nn.Flatten(),
#             #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(832,512),nn.ReLU(),nn.Dropout(0.4),
#             nn.Linear(512,n_classes),
#         )

#     def forward (self, x: torch.Tensor)->torch.Tensor:
#         # print(x.size())
#         x=self.features(x)
#         # print(x.size())
#         # # x=x.view(x.size(0),-1)
#         # print(x.size())
#         x=self.classifier(x)
#         # print(x.size())
#         return x


# model = Model(8,3)
# init_weights = torch.rand([2, 8,60])
# print (model(init_weights))
# exit()

##################################### HYPERPARAMETERS ##########################################################

LR = 0.95e-4 #~5e-5                        new# 1e-4(49.80%)nul, 5e-5(pas mal), 7e-5(50.55%), 5e-4,
EPOCH = 15
BATCH_SIZE = 128
FRACTION = 1            #value between 0-1
NUM_WORKERS = 10
DECAY = 1e-4

##################################### DATA-EMBEDDING ##########################################################

def create_data(starting_dir: str="data") -> int :
    ACTIONS = ["left", "right", "none"]
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]
        training_data[action] = np.array(training_data[action])[:,0:8]  #Adapt The Data to our numbers of electrodes on the EEG

    # creating X, y 
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:
            if action == "left":
                combined_data.append([data, 0])
            elif action == "right":
                combined_data.append([data, 2])
            elif action == "none":
                combined_data.append([data, 1])

    np.random.shuffle(combined_data)
    X=[]
    y=[]
    for a,b in combined_data:
        X.append(a)
        y.append(b)

    X , y = np.array(X), np.array(y)
    if starting_dir == "data":
        np.save("TrainingX",X[:int(len(X)*FRACTION)])
        np.save("Trainingy",y[:int(len(X)*FRACTION)])
    else : 
        np.save("TestingX",X[:int(len(X)*FRACTION)])
        np.save("Testingy",y[:int(len(X)*FRACTION)])

    # print(" Data created succesfully ")
    return int(len(X)*FRACTION)

# def pick_data(dir: str="Training",idx: int=0) -> Tuple[torch.Tensor,int]:
#     X , y = np.load(str(dir)+"X.npy",mmap_mode="r")[idx],np.load(str(dir)+"y.npy",mmap_mode="r")[idx]    #Creating TrainingX.npy and Trainingy.npy (or Testing) files
#     return (torch.from_numpy(np.array(X)).float(),int(y))                                                #    for picking random arrays from them

##################################### MODEL SETUP ##########################################################

lenght_train = create_data("data")
lenght_test = create_data("validation_data")

X_TRAIN, Y_TRAIN = np.load("TrainingX.npy",mmap_mode="r"),np.load("Trainingy.npy",mmap_mode="r")
X_TEST, Y_TEST = np.load("TestingX.npy",mmap_mode="r"),np.load("Testingy.npy",mmap_mode="r")

train_dataset = EEGData(np.array(X_TRAIN),np.array(Y_TRAIN))
test_dataset = EEGData(np.array(X_TEST),np.array(Y_TEST))
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True, num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False, num_workers=NUM_WORKERS,pin_memory=True ,drop_last=True)

# train_dataset = EEGData("Training",lenght_train)
# test_dataset = EEGData("Testing",lenght_test)
# trainloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS,pin_memory = True)
# testloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS,pin_memory = True )

model = Model().cuda()

criterion = nn.CrossEntropyLoss().cuda()
opt = AdamW(model.parameters(),lr=LR ,weight_decay=DECAY)


####################################### TRAINING ###################################################################

train_accuracy=[]
test_accuracy=[]
train_loss=[]
test_loss=[]

for epoch in tqdm(range(EPOCH),desc="Epoch"):
    model.train()
    with tqdm(trainloader, desc="Train") as pbar:
        total_loss = 0
        acc = 0
        for input,label in pbar:
            input,label = input.cuda(), label.cuda()
            opt.zero_grad()
            output = model(input)
            loss = criterion(output,label)
            loss.backward()
            opt.step()
            acc += (torch.argmax(output, dim = 1) == label).sum()/len(train_dataset)
            total_loss += loss.item()/len(trainloader)
            pbar.set_postfix(loss = total_loss,acc = f"{acc*100:.2f}%")

    train_accuracy.append(torch.Tensor.cpu(acc))
    train_loss.append(total_loss)
    model.eval()
    with tqdm(testloader,desc="Valid") as pbar:
        with torch.no_grad():
            total_loss = 0
            acc= 0

            for input,label in pbar:
                input,label = input.cuda() ,label.cuda()
                opt.zero_grad()
                output = model(input)
                loss = criterion(output,label)
                acc += (torch.argmax(output, dim = 1) == label).sum()/len(test_dataset)
                total_loss += loss.item()/len(testloader)            
                pbar.set_postfix(loss = total_loss,acc = f"{acc*100:.2f}%")
    test_accuracy.append(torch.Tensor.cpu(acc))
    test_loss.append(total_loss)
    torch.save(model.state_dict(), f"model/E_{epoch:02d}-A_{acc*100:.2f}%-LR_{LR}.pth")


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([k for k in range(EPOCH)], train_accuracy, color='blue', label="Train")
ax1.plot([k for k in range(EPOCH)], test_accuracy, color='red', label="Test")
ax2.plot([k for k in range(EPOCH)], train_loss, color='blue', label="Train")
ax2.plot([k for k in range(EPOCH)], test_loss, color='red', label="Test")
ax1.legend(facecolor='white')
ax2.legend(facecolor='white')
ax1.set(xlabel="nombre Epoch", ylabel="Accuracy")
ax2.set(xlabel="nombre Epoch", ylabel="Loss")
fig.canvas.draw()
plt.show()


############################# TESTING THE MODEL #############################################


# model = Model().cuda()
# model.load_state_dict(torch.load("best_model/classical/E_02-A_50.03%-LR_7e-05.pth"))
# model.eval()

# preds=[]
# labels=[]

# with torch.no_grad():
#             for input,label in testloader:
#                 input,label = input.cuda() ,label.cuda()
#                 opt.zero_grad()
#                 output = model(input)
#                 pred=torch.zeros((len(output)))

#                 for k in range(len(output)):
#                     pred[k]=torch.argmax(output[k])
#                 if preds==[]:
#                     preds=pred
#                     labels=label
#                 else:
#                     preds= torch.cat((preds,pred),0)
#                     labels= torch.cat((labels,label),0)


# conf = confusion_matrix(labels.cpu().data.numpy(), preds.cpu().data.numpy())
# conf =conf.astype('float')
# for i in range(len(conf)):
#     ligne=0
#     for j in range(len(conf[i])):
#         ligne+=conf[i][j]
    
#     conf[i]=[conf[i][k]/ligne for k in range(len(conf[i]))]

# categorie=["left","none","right"]

# sn.heatmap(conf,annot=True,xticklabels=categorie,yticklabels=categorie,cmap="Blues",fmt=".2%")
# plt.title("Model 49.33% Accuracy")   #BEST :47.82, 
# plt.xlabel("Action Thought")
# plt.ylabel("Predicted Action")
# plt.show()