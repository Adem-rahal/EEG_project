from pylsl import StreamInlet, resolve_stream
from collections import deque, Counter

import torch.nn as nn
import numpy as np
import time
import cv2
import os
import torch




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

last_print = time.time()
fps_counter = deque(maxlen=150)

model = Model()
model.load_state_dict(torch.load("best_model/classical/E_02-A_49.34%-L_1.023.pth"))
model.eval()

FFT_MAX_HZ = 60

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

action = deque(maxlen=20)

while True:  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(8): # each of the 8 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    print(f"time from last fram: { time.time() - last_print}")
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)

    with torch.no_grad():
        network_input = torch.tensor([channel_data])
        out = model(network_input)
        out = torch.softmax(out,dim=1)
        action.append(torch.argmax(out))
        
    a = Counter(action).most_common(1)
    
    if a[0][0] == 0 :
        print("LEFT")
    elif a[0][0] == 1:
        print("NONE")
    else:
        print("RIGHT")