# EEG_project Quantum vs Classical
 


Classical Computing vs Quantum Computing 

image here


The objectif here is to try to classify thoughts using the OpenBCI EEG with a classical machine learning algorithm, a quantum machine learning and an hybrid one.

You can check out OpenBCI's products here: https://shop.openbci.com/

The data set used come from sentdex tutorial on the OpenBCI that you can find here: https://github.com/Sentdex/BCI

I’ve started to work with what he has already done but I had to translate everything to pytorch and try to improve the model, so I advise to read his ReadMe.md file to get a sense of what we are working with.

In my case I only had 8 electrodes on my headset so the following results will be on the 8 first channels, even if the data set does have up to 16 channels available.


 # Quantum model

We will be working with pennylane for the quantum machine learning framework

The idea here was to do a simple classifier using a simple VC (Variational classifier) and see if we can get something from it. Knowing we only have to work on 8 channels we will be using 8 qubits to embed the data it is not optimal but it was the main idea here. We have 60 FFT amplitudes value per qubit so we will repeat the embedding operation 60 times on each qubit. 

If you have a better way to embed the data feel free to try and send me your results I’ll be glad to see what I could have done better.

In the beginning I wanted to put a strongly entangled layer after each step of embedding but the cost was way too high. To optimize the time I was forced to change my idea and only put a basic entangled layer at the end of the circuit which is really not optimal but I was able to earn a lot of time with this.

But it wasn’t enough to train the actual quantum model that I had. Indeed it takes 12h/epoch for the forward and backward pass on all the data set which is way too long for such a simple model. The issue here is the optimization it takes a lot of trials and error to optimize our hyper-parameters and with 12h/epoch it would takes month to have a decent model.

We can clearly see the actual limits of quantum computing today (march 2021). We are too limited in terms of computing power, without access to proper hardware this type of task will remain impossible in a decent amount of time. So if you’re reading this in the futur and you have access to a quantum computer you can try and see if it’s my model or me that sucks.

# Classical model 

I have used here a simple CONVNET using Conv1d because we are working with fourier transform values. Here is the circuit layout:

Conv1d(64, kernel=3, stride=1), ReLu, MaxPool1d(2), Dropout(0.4)
Conv1d(128, kernel=2, stride=2), ReLu, MaxPool1d(2)
Conv1d(256,kernel=2,stride=2), ReLu, MaxPool1d(2), Dropout(0.4)
Flatten()
Linear(512), ReLu, Dropout(0.4)
Linear(3)


As you can see I haven’t used Batch normalization here because I’m still not sure why but with different trial I’ve managed to isolate the issue, it seems like batch normalization was causing gradient explosion and removing them seems to have solved it. The training is more stable like this but maybe I’m wrong so I advise you to try for yourself.

Here is my best model on the validation set: 


image here


# Hybrid model

This last model I was able to make it work in a reasonable amount of time 1h/epoch for the full data set training but I had a week to find an acceptable model which I thought I had. I’ve managed to pull 40% of Accuracy on testing but when I plotted the confusion matrix the results where awful, indeed the model was bias towards one of the output every time maybe my model is fundamentally wrong I don’t know. I didn’t have enough time to work on this so if you want to continue and improve what I have done feel free.


# Conclusion 

In march 2021 the classical machine learning is still the thing to go for if you want to do something practical, it seems that I was a bit early in the quantum wave.

I would be glad to discuss your ideas/work to improve what I have done I’m no pro I’m still learning.

