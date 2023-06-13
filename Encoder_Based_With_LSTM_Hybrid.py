import os

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import numpy as np
import torch
import torch
import os
from os.path import dirname, join as pjoin
from scipy.io import wavfile
import scipy.io
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch import nn, Tensor
import math

import torchaudio.transforms as transforms
from torchaudio import transforms as transformsA
from torchvision import transforms as transforms
from torch.utils.data import Dataset, DataLoader
import math
import wandb

weights = dict()
best_param = dict()
best_acc = 0
best_loss = 100

class AudioDataset(Dataset):
  def __init__(self, parent_directory, window_size = 0.015, frequency = 16000,  transform=None):

    self.parent_directory = parent_directory
    self.transform = transform
    self.window_size = window_size * frequency

    self.audio_files = self._load_input_paths()
    self.labels = self._load_labels()

  def __len__(self):
    return len(self.audio_files)

  def _load_labels(self):
    label_file_paths = []
    for x, y in enumerate(os.listdir(self.parent_directory)):
      if y[-4:] == ".txt":
        if y[-11:] == 'dialect.txt':
          label_file_paths.append(self.parent_directory+y)
    open_files = [open(y, 'r') for x, y in enumerate(label_file_paths)]
    labels = [y.readlines() for x, y in enumerate(open_files)]
    for x, y in enumerate(open_files):
        y.close()
    return labels

  def _load_input_paths(self):
    input_paths = [self.parent_directory + y for x, y in enumerate(os.listdir(self.parent_directory)) if y[-4:] == '.wav']
    return input_paths

  def __getitem__(self, idx):
    file_path = self.audio_files[idx]
    waveform, sample_rate = torchaudio.load(file_path)

    windows = self._split_audio(waveform)

    if self.transform is not None:
      windows = [self.transform(window) for window in windows]

    windows_with_pos = [self._add_positional_encoding(window) for window in windows]

    if len(self.labels) != 0:
      label = self.labels[idx]
      label_tensor = torch.tensor([int(l) for l in label])
      return torch.stack(windows_with_pos), label_tensor
    else:
      return torch.stack(windows_with_pos)
  

  def collate_fn(self, batch):
    # zipped_batch = list(zip(*batch))
    # print(len(batch))
    # print(len(batch[0]))
    # print(len(zipped_batch))
    if isinstance(batch[0], tuple):
      windows, labels = zip(*batch)
      
      windows = torch.stack(windows)
      windows = torch.reshape(windows, (windows.size()[0], windows.size()[2], -1))
      # print(windows.size())
      labels = torch.stack(labels).flatten()
      return windows, labels
    else:
      windows = batch

      windows = torch.stack(windows)
      windows = torch.reshape(windows, (windows.size()[0], windows.size()[2], -1))
      # windows = torch.reshape(windows, (len(windows), len(windows[0][0]), -1))
      # print(windows.size())

      return windows

      
  
  def _split_audio(self, waveform):
    num_samples = waveform.size(1)
    num_windows = int(num_samples // self.window_size)
    window_size = int(self.window_size)

    unfold = waveform.unfold(1, window_size, window_size)

    windows = unfold.transpose(0,1).reshape(-1, unfold.size(0), window_size)
    max_length = 2000
    padded_windows = []
    padding = torch.zeros((abs(max_length - windows.shape[0]), 1, window_size))
    padded_window = torch.cat([windows, padding], dim=0)
    padded_windows.append(padded_window)
    return padded_windows

  def _add_positional_encoding(self, window):
    window_res = torch.reshape(window, (window.size()[0], window.size()[2], window.size()[3])) 
    num_dims, window_size = window_res[0].size()

    pos = torch.arange(0, window_size//2).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, window_size,2) * -(math.log(10000.0)/window_size))

    pos_encoding = torch.zeros(num_dims, window_size)
    pos_encoding[0,0::2] = torch.sin(pos * div_term)
    pos_encoding[0,1::2] = torch.cos(pos * div_term)

    window = torch.cat([window_res[0],pos_encoding], dim=0)

    return window

class LSTMEncoderTransformer(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, num_heads, dropout =0.1):
    super(LSTMEncoderTransformer, self).__init__()
    
    self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)

    self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)

    self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=hidden_size*4, dropout=dropout),
                                                     num_layers=num_layers)
    

    self.Linear = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    lstm_output, _ = self.LSTM(x)
    transformer_output = self.transformer_encoder(lstm_output)
    attention_output, _= self.attention(transformer_output, transformer_output, transformer_output)

    output = self.Linear(attention_output[:,-1,:])

    return output


def manuel():
  batch_size = 32
  frequency = 16000
  window_size = 0.001875 #
  n_fft = int(window_size*frequency)
  hop_length = 128
  num_tokens = n_fft #size of waveform data
  num_features = n_fft#size of waveform data
  input_size = n_fft
  hidden_size = 16
  num_layers = 8
  num_classes = 8
  num_heads = 8
  dropout = 0.70312436924161
  learning_rate = 0.012108793724780052
  num_epochs = 4

  parent_directory_train = '/academic/CSCI481/202320/project_data/task2/train/'
  parent_directory_dev = '/academic/CSCI481/202320/project_data/task2/dev/'

  transform = transforms.Compose([transformsA.Spectrogram(n_fft=n_fft,
                                                      hop_length = hop_length,
                                                      window_fn=torch.hann_window,
                                                      win_length=int(window_size*frequency),
                                                      pad=10000),
                              transforms.Resize((num_tokens, num_features), antialias=True)])
  torch.set_default_dtype(torch.float32)
  # print(torch.cuda.memory_summary())
  
  dataset = AudioDataset(parent_directory_train, window_size = window_size, frequency= frequency, transform = transform)
  dev_dataset = AudioDataset(parent_directory_dev, window_size = window_size, frequency= frequency, transform = transform)
  train_size = int(0.8*len(dataset))
  val_size = len(dataset)-train_size

  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=4)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers = 4)
  dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers = 4)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = LSTMEncoderTransformer(input_size, hidden_size, num_layers, num_classes, num_heads, dropout)

  m = model.to(device)

  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
  # wandb.watch(model, log='all')

  for epoch in range(num_epochs):
    m.train()
    print(f"epoch: {epoch}")
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = m(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    m.eval()
    val_loss = 0.0
    val_correct = 0
    total_samples = 0
    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = m(inputs)

        _, predicted = torch.max(outputs, dim=1)
        val_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loss = criterion(outputs, labels)
        val_loss += loss.item()*inputs.size(0)

    val_accuracy = val_correct / total_samples
    val_loss /= len(dev_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}]")
    global best_acc
    global best_loss
    global weights
    global best_param
    if val_accuracy > best_acc or val_loss < best_loss:
      if val_accuracy > best_acc:
        best_acc = val_accuracy
      if val_loss < best_loss:
        best_loss = val_loss
        
      weights = m.state_dict()
      best_param['batch_size'] =    batch_size
      best_param['num_heads'] =     num_heads
      best_param['num_layers'] =    num_layers
      best_param['dropout'] =       dropout
      best_param['learning_rate'] = learning_rate
      best_param['window_size'] =   window_size
      
  m.eval()
  dev_loss = 0.0
  dev_correct = 0
  total_samples = 0
  with torch.no_grad():
    for inputs, labels in dev_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = m(inputs)

      _, predicted = torch.max(outputs, dim=1)
      dev_correct += (predicted == labels).sum().item()
      total_samples += labels.size(0)

      loss = criterion(outputs, labels)
      dev_loss += loss.item()*inputs.size(0)

  dev_accuracy = dev_correct / total_samples
  dev_loss /= len(dev_dataset)

  print(f"dev Loss: {dev_loss:.4f}, dev Accuracy: {dev_accuracy:.4f}]")

  torch.save(weights, './DL/FinalProj/weights/model_weights2')
    
  

#--------------------------------WANDB-------------------------------

def train():
  print("starting epochs")

  wandb.init(config={'batch_size'    : 32,
                     'num_heads'     : 4,
                     'num_layers'    : 4,
                     'hidden_size'   : 16,
                     'dropout'       : 0.1,
                     "learning_rate" : 0.01,
                     'window_size'   : 0.001875})
  
  global config
  config = wandb.config

  frequency = 16000
  n_fft = int(config.window_size*frequency)
  input_size = n_fft
  num_tokens = n_fft #size of waveform data
  num_features = n_fft#size of waveform data
  
  hop_length = 128
  num_classes = 8
  num_epochs = 4
  
  parent_directory_train = '/academic/CSCI481/202320/project_data/task2/train/'
  parent_directory_dev = '/academic/CSCI481/202320/project_data/task2/dev/'
  transform = transforms.Compose([transformsA.Spectrogram(n_fft=n_fft,
                                                      hop_length = hop_length,
                                                      window_fn=torch.hann_window,
                                                      win_length=int(config.window_size*frequency),
                                                      pad=10000),
                              transforms.Resize((num_tokens, num_features), antialias=True)])
  torch.set_default_dtype(torch.float32)
  dataset = AudioDataset(parent_directory_train, window_size = config.window_size, frequency= frequency, transform = transform)
  dev_dataset = AudioDataset(parent_directory_dev, window_size = config.window_size, frequency= frequency, transform = transform)

  train_size = int(0.8*len(dataset))
  val_size = len(dataset)-train_size

  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
  dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = LSTMEncoderTransformer(input_size, config.hidden_size, config.num_layers, num_classes, config.num_heads, config.dropout)

  m = model.to(device)

  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.Adam(m.parameters(), lr=config.learning_rate)
  wandb.watch(m, log='all')

  for epoch in range(num_epochs):
    m.train()
    for inputs, labels in train_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = m(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    m.eval()
    val_loss = 0.0
    val_correct = 0
    total_samples = 0
    with torch.no_grad():
      for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = m(inputs)

        _, predicted = torch.max(outputs, dim=1)
        val_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loss = criterion(outputs, labels)
        val_loss += loss.item()*inputs.size(0)

    val_accuracy = val_correct / total_samples
    val_loss /= len(val_dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}]")
    global best_acc
    global best_loss
    global weights
    global best_param
    if val_accuracy > best_acc or val_loss < best_loss:
      if val_accuracy > best_acc:
        best_acc = val_accuracy
      if val_loss < best_loss:
        best_loss = val_loss
        
      weights = m.state_dict()
      best_param['batch_size'] = config.batch_size
      best_param['num_heads'] = config.num_heads
      best_param['num_layers'] = config.num_layers
      best_param['dropout'] = config.dropout
      best_param['learning_rate'] = config.learning_rate
      best_param['window_size'] = config.window_size

    m.eval()
    dev_loss = 0.0
    dev_correct = 0
    total_samples = 0
    with torch.no_grad():
      for inputs, labels in dev_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = m(inputs)

        _, predicted = torch.max(outputs, dim=1)
        dev_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loss = criterion(outputs, labels)
        dev_loss += loss.item()*inputs.size(0)

    dev_accuracy = dev_correct / total_samples
    dev_loss /= len(dev_dataset)

  print(f"dev Loss: {dev_loss:.4f}, dev Accuracy: {dev_accuracy:.4f}]")

  wandb.log({'dev_acc': dev_accuracy, 'dev_loss':dev_loss, 'val_accuracy':val_accuracy, 'val_loss':val_loss})
  file_name = "model_weights" + str(dev_accuracy)
  torch.save(weights, './DL/FinalProj/model_weights/'+file_name )

def load_model():
  batch_size = 1
  frequency = 16000
  window_size = 0.001875 #
  n_fft = int(window_size*frequency)
  hop_length = 128
  num_tokens = n_fft #size of waveform data
  num_features = n_fft#size of waveform data
  input_size = n_fft
  hidden_size = 16
  num_layers = 8
  num_classes = 8
  num_heads = 8
  dropout = 0.70312436924161
  learning_rate = 0.012108793724780052
  num_epochs = 4

  parent_directory_train = '/academic/CSCI481/202320/project_data/task2/train/'
  parent_directory_dev = '/academic/CSCI481/202320/project_data/task2/dev/'
  test_directory = '/academic/CSCI481/202320/project_data/task2/test/'
  
  transform = transforms.Compose([transformsA.Spectrogram(n_fft=n_fft,
                                                      hop_length = hop_length,
                                                      window_fn=torch.hann_window,
                                                      win_length=int(window_size*frequency),
                                                      pad=10000),
                              transforms.Resize((num_tokens, num_features), antialias=True)])
  torch.set_default_dtype(torch.float32)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  test_dataset = AudioDataset(test_directory, window_size = window_size, frequency= frequency, transform = transform)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

  model = LSTMEncoderTransformer(input_size, hidden_size, num_layers, num_classes, num_heads, dropout)
  criterion = nn.CrossEntropyLoss()

  m = model.to(device)
  optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
  
  m.load_state_dict(torch.load('/home/osugas2/model_weights'))
  
  m.eval()
  
  dev_loss = 0.0
  dev_correct = 0
  total_samples = 0
  output_cats = torch.empty((0,8), dtype=torch.float32)
  output_cats = output_cats.to(device)
  with torch.no_grad():
    for inputs in test_loader: #, labels
      inputs = inputs.to(device)
      # labels = labels.to(device)

      output_cats = torch.cat((output_cats, m(inputs)), 0)
      
  #     _, predicted = torch.max(outputs, dim=1)
  #     dev_correct += (predicted == labels).sum().item()
  #     total_samples += labels.size(0)

  #     loss = criterion(outputs, labels)
  #     dev_loss += loss.item()*inputs.size(0)

  # dev_accuracy = dev_correct / total_samples
  # dev_loss /= len(test_dataset)

  # print(f"dev Loss: {dev_loss:.4f}, dev Accuracy: {dev_accuracy:.4f}]")
  print(output_cats)
  output_cats = output_cats.cpu()
  # np.save('task2_predictions', outputs)
  # torch.save(output_cats.numpy(), 'task2_predictions')
  np.save('task2_predictions.npy', output_cats.numpy())
  

def main():
      
    sweep_config = {
        'method' : 'bayes', #grid, random, bayesian
    }

    metric = {
        'name' : 'Dev Loss',
        'goal' : 'minimize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {

        'learning_rate' : {
            'max' : 0.2,
            'min' : 0.0001,
        },

        'batch_size' : {
            'values' : [32]
        },
        
        'num_layers' : {
            'values' : [8]
        },
        'num_heads' : {
            'values' : [8]
        },
        'dropout' : {
            'max' : 0.5,
            'min' : 0.0 
        },
        'window_size' : {
            'values' : [0.001875]
        },
        'hidden_size': {
            'values' : [8]
        }
    }

    sweep_config['parameters'] = parameters_dict
    
    
    

    
    

    
    sweep_id = wandb.sweep(sweep_config, project='lab')
    
    
    train_fn = lambda : train()
    wandb.agent(sweep_id,function=train_fn, count=20)



if __name__ == "__main__":
    # main()
    # manuel()
    load_model()