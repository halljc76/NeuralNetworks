#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import HTML
import pandas as pd


# In[2]:


train = pd.read_csv("./input/digit-recognizer/train.csv")


# In[3]:


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Get the device we're training on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_digits(df):
    """Loads images as PyTorch tensors"""
    # Load the labels if they exist 
    # (they wont for the testing data)
    labels = []
    start_inx = 0
    if 'label' in df.columns:
        labels = [v for v in df.label.values]
        start_inx = 1
        
    # Load the digit information
    digits = []
    for i in range(df.pixel0.size):
        digit = df.iloc[i].astype(float).values[start_inx:]
        digit = np.reshape(digit, (28,28))
        digit = transform(digit).type('torch.FloatTensor')
        if len(labels) > 0:
            digits.append([digit, labels[i]])
        else:
            digits.append(digit)

    return digits

# Load the training data
train_X = get_digits(train)

# Some configuration parameters
num_workers = 0    # number of subprocesses to use for data loading
batch_size  = 64   # how many samples per batch to load
valid_size  = 0.2  # percentage of training set to use as validation

# Obtain training indices that will be used for validation
num_train = len(train_X)
indices   = list(range(num_train))
np.random.shuffle(indices)
split     = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
from torch.utils.data.sampler import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Construct the data loaders
trainLoad = torch.utils.data.DataLoader(train_X, batch_size=batch_size,
                    sampler=train_sampler, num_workers=num_workers)
testLoad = torch.utils.data.DataLoader(train_X, batch_size=batch_size, 
                    sampler=valid_sampler, num_workers=num_workers)

# Test the size and shape of the output
dataiter = iter(trainLoad)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# In[4]:


trainSet = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
testSet = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
# trainLoad = torch.utils.data.DataLoader(dataset=trainSet, batch_size=5, shuffle=True)
# testLoad = torch.utils.data.DataLoader(dataset=testSet, batch_size=5, shuffle=False)


# In[5]:


classes = (i for i in range(10))

def showImg(image):
    image = (image / 2) + 0.5
    npImage = image.numpy()
    plt.imshow(np.transpose(npImage, (1, 2, 0))) # Throws a warning, though it 
                                                 # can be overlooked for what we're
                                                 # doing here.
    plt.show()

iterator = iter(trainLoad)
imgs, names = iterator.next()
print(imgs.shape, names.shape)

showImg(torchvision.utils.make_grid(imgs[:5]))
string = " "

for i in range(5):
    string += (str(names[i].item()) + " ")
print(string)


# In[6]:


class myModel(nn.Module):
    
    def __init__(self):
        super(myModel, self).__init__()
        self.convNet1 = nn.Conv2d(1, 15, kernel_size = 3)
        self.convNet2 = nn.Conv2d(15, 30, kernel_size = 3)
        self.maxPool = nn.MaxPool2d(2)
        self.linear = nn.Linear(5 * 5 * 30, 15)
        self.linear2 = nn.Linear(15, 10)
        
    def forward(self, x):
        inputSize = x.size(0)
        x = F.relu(self.maxPool(self.convNet1(x)))
        x = F.relu(self.maxPool(self.convNet2(x)))
        x = x.view(inputSize, -1)
        x = F.softmax(self.linear(x), dim = -1)
        x = self.linear2(x)
        return x
    
class myModel2(nn.Module):
    
    def __init__(self):
        super(myModel2, self).__init__()
        
        # The two Convolutional-Pooling Sets of Layers, 
        # connected through RELU functions.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size = 5, 
                      padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, 
                         stride = 3)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size = 5, 
                      padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, 
                         stride = 3)
            )
        
        # Dropout 
        self.dropoutLayer = nn.Dropout()
        
        # Linear Layers
        self.linear1 = nn.Linear(4 * 4 * 60, 1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 250)
        self.linear3 = nn.Linear(500, 10)
    
    def forward(self, x):
        
        # Feed-forward network function
        net = self.layer1(x)
        net = self.layer2(net)
        net = net.reshape(net.size(0), -1) 
        net = self.dropoutLayer(net)
        net = self.linear1(net)
        net = self.linear2(net)
        
        return net
        

CNN = myModel()
CNN2 = myModel2()

lossFunc = nn.CrossEntropyLoss()
lossFunc = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN.parameters(), lr = 0.001, momentum = 0.4)
optimizer2 = optim.Adam(CNN2.parameters(), lr = 0.0001)

epochList = [i for i in range(1,20)]
totalLosses = []
totalLosses2 = []

for epoch in range(len(epochList)):
    lossAtEpoch = 0.0
    lossAtEpoch2 = 0.0
   
    
    for index, data in enumerate(trainLoad, start = 0):
        inputs, names = data
        
        optimizer.zero_grad() # Zero the param gradients
        optimizer2.zero_grad()
        
        outputs = CNN(inputs)
        outputs2 = CNN2(inputs)
        
        
        loss = lossFunc(outputs, names)
        loss2 = lossFunc(outputs2, names)
        
        
        loss.backward() # Make loss into Tensor
        loss2.backward()
        
        optimizer.step()
        optimizer2.step()
        
        
        lossAtEpoch += loss.item()
        lossAtEpoch2 += loss2.item()
        
        
        mini_batch_size = 250
        if index % mini_batch_size == (mini_batch_size - 1):
            print("Loss: " + str(lossAtEpoch / mini_batch_size)[:5])
            totalLosses.append(lossAtEpoch / mini_batch_size)
            
            print("Loss Model 2: " + str(lossAtEpoch2 / mini_batch_size)[:5])
            totalLosses2.append(lossAtEpoch2 / mini_batch_size)
           
            lossAtEpoch = 0.0
            lossAtEpoch2 = 0.0
        


# In[7]:


arrayOfLosses = np.array(totalLosses)
arrayOfLosses2 = np.array(totalLosses2)


# In[8]:


arrayOfLosses.shape # =(38,0)
arrayOfLosses = arrayOfLosses.reshape(2,19)

arrayOfLosses2 = arrayOfLosses2.reshape(2,19)


# In[9]:


df = pd.DataFrame({"Batch No.": [i for i in range(1,2 * arrayOfLosses.shape[1] + 1)],
                   "Loss (Model 1)": totalLosses})
df.reset_index()
df.to_excel("lossData.xlsx")

df2 = pd.DataFrame({"Loss (Model 2)": totalLosses2})
df2.reset_index()
df2.to_excel("lossData2.xlsx")


# In[10]:


plt.plot(list(arrayOfLosses[0]), label = "Model 1, Batch 1 - 19")
plt.plot(list(arrayOfLosses[1]), label = "Model 1, Batch 20 - 38")
plt.xlabel("x-th Mini Batch (N = 4000)")
plt.ylabel("Loss Function (CrossEntropyLoss)")
# plt.xticks(np.arange(len(list(arrayOfLosses[0]))), np.arange(1, len(arrayOfLosses[0])+1), rotation = 'vertical')
plt.legend()
plt.title("Loss Function of CNN with MNIST Dataset ")

plt.plot(list(arrayOfLosses2[0]), label = "Model 2, Batch 1 - 19")
plt.plot(list(arrayOfLosses2[1]), label = "Model 2, Batch 20 - 38")
plt.legend()
# plt.xticks(np.arange(len(list(arrayOfLosses2[0]))), np.arange(1, len(arrayOfLosses2[0])+1), rotation = 'vertical')
plt.show()


# In[11]:


plt.plot(totalLosses, label = "Model 1")
plt.title("Loss Function of CNN with MNIST Dataset")
plt.ylabel("Loss Function (CrossEntropyLoss)")
plt.xlabel("x-th Mini Batch (N = 4000)")
plt.legend()

plt.plot(totalLosses2, label = "Model 2")
plt.legend()
plt.show()


# In[12]:


HTML(pd.concat([df, df2], axis = 1).to_html(index=False, classes = "dataframe"))


# In[13]:


numCorrect = list(0. for i in range(10))
numTotal = list(0. for i in range(10))
with torch.no_grad():
    for data in trainLoad:
        images, labels = data
        outputs = CNN(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            numCorrect[label] += c[i].item()
            numTotal[label] += 1

avg = 0

for i in range(10):
    avg += 100 * numCorrect[i] / numTotal[i]
    print('Accuracy of %5s : %2d %%' % (
        i + 1, 100 * numCorrect[i] / numTotal[i]))

print(avg / 10)


# In[14]:


numCorrect = list(0. for i in range(10))
numTotal = list(0. for i in range(10))
with torch.no_grad():
    for data in trainLoad:
        images, labels = data
        outputs = CNN2(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            numCorrect[label] += c[i].item()
            numTotal[label] += 1

avg = 0

for i in range(10):
    avg += 100 * numCorrect[i] / numTotal[i]
    print('Accuracy of %5s : %2d %%' % (
        i + 1, 100 * numCorrect[i] / numTotal[i]))

print(avg / 10)


# In[ ]:





# In[15]:


ImageId, Label = [], []

for index, data in enumerate(testLoad):
    images, labels = data
    
    outputs = CNN2(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    
    for i in range(len(predicted)):
        ImageId.append(len(ImageId) + 1)
        Label.append(predicted[i].numpy())

sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})
sub.describe


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




