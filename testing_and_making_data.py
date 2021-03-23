from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import cv2
import os
import random
#import tensorflow as tf
from collections import OrderedDict
import torch

from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.optim as optim




class Model(nn.Module):     #BEST : 49,34% LR=10e-5
    def __init__(self, channels: int=8, n_classes: int=3)->None:
        super(Model,self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm1d(channels),nn.Conv1d(channels,64,kernel_size=3,stride=1),nn.ReLU(),nn.BatchNorm1d(64),nn.Dropout(0.4),
            nn.Conv1d(64,128,kernel_size=2,stride=2),nn.ReLU(),nn.BatchNorm1d(128),nn.MaxPool1d(2),
            nn.Conv1d(128,256,kernel_size=2,stride=2),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.4),
            nn.Flatten(),
            #nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768),nn.Linear(768,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,n_classes),
        )

    def forward (self, x: torch.Tensor)->torch.Tensor:
        # print(x.size())
        x=self.features(x)
        # print(x.size())
        x=self.classifier(x)
        # print(x.size())
        return x

# reshape = (-1, 8, 60)

model = Model()
model.load_state_dict(torch.load("best_model/classical/E_02-A_49.34%-L_1.023.pth"))
model.eval()

test = torch.rand((32,8,60))


reshape = (-1, 8, 60)
'''

model = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
print(type(model))
reshape = (-1, 8, 60)
#predicton = model( np.zeros((32,8,60)).reshape(reshape) )
prediction = model.eval(np.zeros((32,8,60)).reshape(reshape))
'''
ACTION = 'left' # THIS IS THE ACTION YOU'RE THINKING 

FFT_MAX_HZ = 60

HM_SECONDS = 10  # this is approximate. Not 100%. do not depend on this.
TOTAL_ITERS = HM_SECONDS*25  # ~25 iters/sec
BOX_MOVE = "model"  # random or model

last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

WIDTH = 800
HEIGHT = 800
SQ_SIZE = 50
MOVE_SPEED = 1

square = {'x1': int(int(WIDTH)/2-int(SQ_SIZE/2)), 
          'x2': int(int(WIDTH)/2+int(SQ_SIZE/2)),
          'y1': int(int(HEIGHT)/2-int(SQ_SIZE/2)),
          'y2': int(int(HEIGHT)/2+int(SQ_SIZE/2))}


box = np.ones((square['y2']-square['y1'], square['x2']-square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

total = 0
left = 0
right = 0
none = 0
correct = 0 

channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(8): # each of the 16 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)

    env = np.zeros((WIDTH, HEIGHT, 3))

    env[:,HEIGHT//2-5:HEIGHT//2+5,:] = horizontal_line
    env[WIDTH//2-5:WIDTH//2+5,:,:] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2']] = box


    cv2.imshow('', env)
    cv2.waitKey(1)


    with torch.no_grad():
        network_input = torch.tensor([channel_data])
        out = model(network_input)
        out = torch.softmax(out,dim=1)
        print(out)
        
        

    if BOX_MOVE == "random":
        move = random.choice([-1,0,1])
        square['x1'] += move
        square['x2'] += move

    elif BOX_MOVE == "model":
        choice = torch.argmax(out)
        print(f"choice, {choice}")
        if choice == 0:
            if ACTION == "left":
                correct += 1
            square['x1'] -= MOVE_SPEED
            square['x2'] -= MOVE_SPEED
            left += 1

        elif choice == 2:
            if ACTION == "right":
                correct += 1
            square['x1'] += MOVE_SPEED
            square['x2'] += MOVE_SPEED
            right += 1

        else:
            if ACTION == "none":
                correct += 1
            none += 1

    total += 1
    channel_datas.append(channel_data)


    
exit()
#plt.plot(channel_datas[0][0])
#plt.show()

datadir = "adem_data"
if not os.path.exists(datadir):
    os.mkdir(datadir)

actiondir = f"{datadir}/{ACTION}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")

for action in ['left', 'right', 'none']:
    #print(f"{action}:{len(os.listdir(f'data/{action}'))}")
    print(action, sum(os.path.getsize(f'data/{action}/{f}') for f in os.listdir(f'data/{action}'))/1_000_000, "MB")

print(ACTION, correct/total)
print(f"left: {left/total}, right: {right/total}, none: {none/total}")

# with open("accuracies.csv", "a") as f:
#     f.write(f"{int(time.time())},{ACTION},{correct/total},{MODEL_NAME},{left/total},{right/total},{none/total}\n")