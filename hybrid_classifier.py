
from collections import Counter
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
from pennylane.init import strong_ent_layers_uniform ,basic_entangler_layers_normal ,basic_entangler_layers_uniform
from pennylane.templates.layers import StronglyEntanglingLayers, BasicEntanglerLayers
from typing import Dict, List, Set, Tuple
from sklearn.metrics import confusion_matrix

import seaborn as sn
import pandas as pd
import sys
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import random
import time


NUM_WIRES = 3
EMBEDDING_LAYERS = 2

################################ DEVICE CHOICE ###########################################

dev = qml.device("default.qubit", wires = NUM_WIRES)  #We are using the default pennylane qubit because it's the most permormant on our model
cuda = torch.device('cuda')

################################ DATASET INITIALIZATION ##################################

class EEGData(Dataset) :
    def __init__(self, x, y, transform=None) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[torch.Tensor,int]:
        return (torch.from_numpy(self.x[idx]).float(),int(self.y[idx]))

######################################## HYBRID MODEL #######################################

def emmbedding_layer(x):
    for i in range(NUM_WIRES):
            qml.RY(x[i]*np.pi,wires=i)   


@qml.qnode(dev, interface="torch",diff_method='adjoint')
def circuit(inputs,weights) -> list(float):
    for k in range(0,15,3):
        qml.templates.AngleEmbedding(inputs[k:k+3],wires = range(NUM_WIRES))
    StronglyEntanglingLayers(init_weights, wires = [k for k in range(NUM_WIRES)])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))


init_weights = Variable(torch.from_numpy(strong_ent_layers_uniform(n_layers=EMBEDDING_LAYERS, n_wires=NUM_WIRES)).float().cuda(),requires_grad=True)  
weights_shape = {"weights": [EMBEDDING_LAYERS, NUM_WIRES, 3] }

''' Lignes to comment for benchmarking simulators speeds '''
        
qlayer = qml.qnn.TorchLayer(circuit, weights_shape)


class Model(nn.Module):
    def __init__(self, channels: int=8, n_classes: int=3)->None:
        super(Model,self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels,64,kernel_size=3,stride=1),nn.ReLU(),nn.Dropout(0.4),
            nn.Conv1d(64,128,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,256,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Flatten(),
        )
        self.hybrid_classifier = nn.Sequential(
            nn.Linear(768,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,15),
            qlayer,
        )

    def forward (self, x: torch.Tensor)->torch.Tensor:
        # print(x.size())
        x=self.features(x)
        # print(x.size())
        x=self.hybrid_classifier(x)
        # print(x.size())
        return x

'''Uncomment these lignes to test your model '''

# model = Model(8,3)
# init_weights = torch.rand([6, 8, 60])
# print (model(init_weights))
# exit()

################################ BENCHMARKING  ####################################

'''To Test how the differents simulators works with your model uncomment this section and comment the Qnode decorator above the circuit 
   method. You will have to install the necessary pennylane plugins yourself according to what device you want to try'''

# test = torch.rand([10, 8, 60])

# weights_shape = { "weights": [EMBEDDING_LAYERS, NUM_WIRES, 3] }
# devices = [
#     qml.device("forest.numpy_wavefunction", wires=NUM_WIRES),
#     qml.device("forest.wavefunction", wires=NUM_WIRES),
#     qml.device("default.qubit", wires=NUM_WIRES),
#     qml.device("qiskit.aer", wires=NUM_WIRES),
#     qml.device("qulacs.simulator", wires=NUM_WIRES ,gpu=True),
#     qml.device("qulacs.simulator", wires=NUM_WIRES ,gpu=False),
#     qml.device('cirq.simulator', wires=NUM_WIRES),
#     qml.device("projectq.simulator", wires=NUM_WIRES),
#     qml.device("qiskit.basicaer", wires=NUM_WIRES),
#     # qml.device("microsoft.QubitSimulator", wires=NUM_WIRES),
#     #qml.device("qiskit.basicaer", wires=NUM_WIRES),
#     # qml.device("forest.qvm", device="{}q-pyqvm".format(NUM_WIRES)),
#     # qml.device("forest.qvm", device="{}q-qvm".format(NUM_WIRES)),
# ]

# print("Encoding size: {}".format([10,8,60]))
# print("Number of qubits: {}".format(NUM_WIRES))

# for i,dev in enumerate(devices):
#     print("\nDevice: {}".format(dev.name))
#     if i == 2:
#         qnode = qml.QNode(circuit, dev,diff_method="adjoint")
#     else:
#         qnode = qml.QNode(circuit, dev)
#     qlayer = qml.qnn.TorchLayer(qnode, weights_shape)
#     model = Model()
#     start = time.perf_counter()       
#     model(test)
#     print(f"Time = {time.perf_counter()-start}")

# exit()

##################################### HYPERPARAMETERS ##########################################################

LR = 7.5e-3  
EPOCH = 6
BATCH_SIZE = 128
FRACTION = 0.5          #value between 0-1
NUM_WORKERS = 8
DECAY = 1e-4

##################################### DATA-EMBEDDING ##########################################################

def create_data(starting_dir="data") -> int :
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
    print(lengths)
    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]
        training_data[action] = np.array(training_data[action])[:,0:8]  #Adapt The Data to your numbers of electrodes on the EEG
                                                                        #    it's 8 in our case

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

    return X[:int(len(X)*FRACTION)], y[:int(len(X)*FRACTION)]    #To not load the full dataset when trying quick changes

#################################### MODEL INITIALISATION ##########################################################

X_TRAIN, Y_TRAIN = create_data("data")                               
X_TEST, Y_TEST = create_data("validation_data")

train_dataset = EEGData(np.array(X_TRAIN),np.array(Y_TRAIN))             
test_dataset = EEGData(np.array(X_TEST),np.array(Y_TEST))

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True ,drop_last=True)

model = Model().cuda()

criterion = nn.CrossEntropyLoss().cuda()
opt = AdamW([init_weights],lr=LR ,weight_decay=DECAY)

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
    torch.save(model.state_dict(), f"model/H_E_{epoch:02d}-A_{acc*100:.2f}%-L_{total_loss:.3f}.pth") #Saving the model at each step


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([k for k in range(EPOCH)], train_accuracy, color='blue', label="Train")
ax1.plot([k for k in range(EPOCH)], test_accuracy, color='red', label="Test")
ax2.plot([k for k in range(EPOCH)], train_loss, color='blue', label="Train")
ax2.plot([k for k in range(EPOCH)], test_loss, color='red', label="Test")
ax1.legend(facecolor='white')
ax2.legend(facecolor='white')
ax1.set(xlabel="Epoch", ylabel="Accuracy")
ax2.set(xlabel="Epoch", ylabel="Loss")
fig.canvas.draw()
plt.show()

############################# TESTING THE MODEL #############################################

# model = Model().cuda()
# model.load_state_dict(torch.load("best_model/hybrid/H_E_00-A_37.00%-L_1.101.pth"))
# model.eval()

# preds = []
# labels = []

# with torch.no_grad():
#             for input,label in testloader:
#                 input,label = input.cuda() ,label.cuda()
#                 opt.zero_grad()
#                 output = model(input)
#                 pred=torch.zeros((len(output)))

#                 for k in range(len(output)):
#                     pred[k] = torch.argmax(output[k])
#                 if preds == []:
#                     preds = pred
#                     labels = label
#                 else:
#                     preds = torch.cat((preds,pred),0)
#                     labels = torch.cat((labels,label),0)

# conf = confusion_matrix(labels.cpu().data.numpy(), preds.cpu().data.numpy())
# conf = conf.astype('float')
# for i in range(len(conf)):
#     ligne = 0
#     for j in range(len(conf[i])):
#         ligne += conf[i][j]
    
#     conf[i] = [conf[i][k]/ligne for k in range(len(conf[i]))]

# categorie = ["left","none","right"]

# sn.heatmap(conf,annot=True,xticklabels=categorie,yticklabels=categorie,cmap="Blues",fmt=".2%")
# plt.title("Model Hybrid xx% Accuracy")   
# plt.xlabel("Action Thought")
# plt.ylabel("Predicted Action")
# plt.show()