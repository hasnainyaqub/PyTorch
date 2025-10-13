#!/usr/bin/env python
# coding: utf-8

# # ANN with PyTorch - Fashion MNIST
# 

# # Import libraries

# In[32]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


# In[30]:


# Set random seed for reproducibility
torch.manual_seed(42)


# In[31]:


# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# # import dataset

# In[7]:


get_ipython().system('unzip /content/fashion_mnist_dataset.zip')


# In[33]:


# import dataset
df = pd.read_csv('/content/fashion-mnist_train.csv')
df.head()


# In[34]:


# let's plot first 10 images with 5x5 grid
fig, ax = plt.subplots(2, 5, figsize=(15, 10))
for i in range(10):
    ax[i//5, i%5].imshow(df.iloc[i, 1:].values.reshape(28, 28), cmap='gray')
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title(df.iloc[i, 0])

plt.tight_layout()
plt.show()


# ## Split the data

# In[35]:


# split the data into features and labels
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X.shape, y.shape


# In[36]:


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ## Scale the data

# In[37]:


# scaling the data
X_train = X_train / 255.0
X_test = X_test / 255.0


# # create dataloader

# In[38]:


# create CustomDataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y


# In[39]:


# create dataloader
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)


# # Model building

# In[47]:


class MyNN(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
        super().__init__()
        layers = []
        for i in range(num_hidden_layers):

            layers.append(nn.Linear(input_dim, neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = neurons_per_layer

        layers.append(nn.Linear(neurons_per_layer, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


# # Objective function

# In[51]:


# Objective function
def objective(trial):
  # next hyperparameter values from the search space
  num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
  neurons_per_layer = trial.suggest_int('neurons_per_layer', 8, 128, step=8)
  epochs = trial.suggest_int('epochs', 10, 100, step=5)
  lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
  dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
  batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
  optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
  weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

  # data loader
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
  test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

  # model init
  input_dim = X_train.shape[1]
  output_dim = 10
  model = MyNN(input_dim, output_dim,num_hidden_layers, neurons_per_layer, dropout_rate)
  model.to(device)

  # optimizer selection
  criterion = nn.CrossEntropyLoss()

  if optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  elif optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

  elif optimizer_name == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

  # training loop
  for epoch in range(epochs):

      for batch_features, batch_labels in train_loader:
          # move data to gpu
          batch_features = batch_features.to(device)
          batch_labels = batch_labels.to(device)

          # forward pass
          outputs = model(batch_features)

          # calculate loss
          loss = criterion(outputs, batch_labels)

          # zero gradients
          optimizer.zero_grad()

          # backward pass
          loss.backward()

          # update weights
          optimizer.step()

  # evaluation
  model.eval()

  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total

  return accuracy


# In[52]:


# lets create study
import optuna
study = optuna.create_study(direction='maximize')


# In[53]:


study.optimize(objective, n_trials=20)


# In[54]:


study.best_value, study.best_params


# In[56]:


# Best trial se model banaen
best_trial = study.best_trial
best_params = best_trial.params
# model init
input_dim = X_train.shape[1]
output_dim = 10
# Model ko train karein best hyperparameters ke saath
model = MyNN(input_dim, output_dim, best_params['num_hidden_layers'], best_params['neurons_per_layer'], best_params['dropout_rate'])
# ... (training code jaisa pehle diya)

# Model ko save karein
torch.save(model.state_dict(), 'best_model.pth')

