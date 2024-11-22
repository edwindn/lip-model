import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

from utils import get_dataloader
#from models import EncoderBatched as Encoder
from models import Encoder


# ---
# PARAMS: batch_size=20, gamma=0.8, epochs=2000
EPOCHS = 200
BATCH_SIZE = 5

LEARNING_RATE = 1e-4
L2_REG = 1e-5
GAMMA = 0.5
DECAY_EVERY = 50

OUTPUT_DIM = 5
DATA_FOLDER = 'ft_data'
INFER_FOLDER = 'ft_data_infer'

LOAD_WEIGHTS = 'encoder_weights_B.pth' # load pre-trained weights
SAVE_WEIGHTS = 'ft_encoder_weights.pth' # save finetuned weights
INFER_WEIGHTS = 'ft_encoder_weights.pth' # load finetuned weights
# ---

WEIGHTS = [0.75, 0.1] # check that the inputs are sequential


def get_arr(txt):
    txt = os.path.join(DATA_FOLDER, txt)
    arr = open(txt, 'r').read()
    arr = ast.literal_eval(arr)
    arr = np.array(arr)
    return arr

def numeric_sort_key(s):
    match = re.search(r'^(\d+)', s)
    return int(match.group(1)) if match else float('inf')

def get_number(file):
    match = re.match(r"(\d+)_", file)
    number = match.group(1)
    return number

def delete_file(folder_path):
    if os.path.exists(folder_path):
        os.remove(folder_path)

def add_noise(inputs, noise_factor):
    noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape)
    return noisy_inputs


# ---
# TRAINING

train_loss_history = []
val_loss_history = []
train_accuracy_history = []
val_accuracy_history = []

def dot(weights, tensors):
  assert all(t.shape == tensors[0].shape for t in tensors), "Not all tensors in the history have the same shape."
  out = torch.zeros_like(tensors[0])
  for w, t, in zip(weights, tensors):
    out += w * t
  return out

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)  # weight_decay adds L2 regularization)
    scheduler = StepLR(optimizer, step_size=DECAY_EVERY, gamma=GAMMA)  # Define the scheduler

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        prev_outputs = [torch.zeros(BATCH_SIZE, OUTPUT_DIM)] * len(WEIGHTS)

        for inputs, labels in train_loader:
            inputs, labels = inputs.squeeze(0), labels.squeeze(0)
            assert inputs.size(0) == BATCH_SIZE, "Inputs size 0 does not match batch size"
            assert labels.size(0) == BATCH_SIZE, "Labels size 0 does not match batch size"

            optimizer.zero_grad()
        
            history_input = dot(WEIGHTS, prev_outputs)
            outputs = model((inputs, history_input))
            loss = criterion(outputs, labels)# + criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            correct_train += (outputs.round() == labels).sum().item()
            total_train += labels.size(0)

            prev_outputs.pop()
            prev_outputs.insert(0, outputs.detach())

        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        prev_outputs = [torch.zeros(BATCH_SIZE, OUTPUT_DIM)] * len(WEIGHTS)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.squeeze(0), labels.squeeze(0)
                assert inputs.size(0) == BATCH_SIZE, "Inputs size 0 does not match batch size"
                assert labels.size(0) == BATCH_SIZE, "Labels size 0 does not match batch size"

                history_input = dot(WEIGHTS, prev_outputs)
                outputs = model((inputs, history_input))
                loss = criterion(outputs, labels)# + criterion(reconstruction, inputs)
                val_loss += loss.item() * inputs.size(0)
                correct_val += (outputs.round() == labels).sum().item()
                total_val += labels.size(0)

                prev_outputs.pop()
                prev_outputs.insert(0, outputs)

        val_loss = val_loss / len(test_loader.dataset)
        val_accuracy = correct_val / total_val
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        current_lr = get_current_lr(optimizer)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}, Val Loss: {val_loss:.4f}')

        scheduler.step()  # Step the scheduler at the end of each epoch

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'loss_{EPOCHS}_{BATCH_SIZE}.png')
    plt.show()

def main_train():
    train_loader, test_loader = get_dataloader(DATA_FOLDER, BATCH_SIZE)

    encoder = Encoder(OUTPUT_DIM)
    encoder.load_state_dict(torch.load(LOAD_WEIGHTS)) # load pre-trained encoder
    train(encoder, train_loader, test_loader)

    torch.save(encoder.state_dict(), SAVE_WEIGHTS)
    print(f'Saved model weights to {SAVE_WEIGHTS}')

def main_infer():
    encoder = Encoder(OUTPUT_DIM)
    encoder.load_state_dict(torch.load(INFER_WEIGHTS))
    encoder.eval()
    print(f'Loaded weights from {INFER_WEIGHTS}')

    print(f'Inferring from {INFER_FOLDER}')
    infer_files = [f for f in os.listdir(INFER_FOLDER) if f.endswith('inputs.txt')]

    for input in infer_files:
        preds = []
        
        path = os.path.join(INFER_FOLDER, input)
        arr = open(path, 'r').read()
        arr = ast.literal_eval(arr)
        arr = np.array(arr)

        prev_preds = [torch.zeros(1, OUTPUT_DIM)] * len(WEIGHTS)
        for frame in arr:
            model_input = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)

            history_input = dot(WEIGHTS, prev_preds)
            pred = encoder((model_input, history_input))
            preds.append(pred.squeeze(0).tolist())
            
            prev_preds.pop()
            prev_preds.insert(0, pred.detach())

        #print(np.array(preds).shape)
        number = get_number(input)
        output_path = os.path.join(INFER_FOLDER, f'{number}_preds.txt')
        with open(output_path, 'w') as f:
            f.write(str(preds))


    files = [f for f in os.listdir(INFER_FOLDER) if f.endswith('landmark_output.txt')]
    for file in files:
        delete_file(os.path.join(INFER_FOLDER, file))
    return


if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'train':
        main_train()
    elif mode == 'infer':
        main_infer()
    else:
        print("\nInvalid mode given. Modes are 'train' or 'infer'")
