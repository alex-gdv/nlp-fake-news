"""
Sources:
https://chriskhanhtran.github.io/posts/cnn-sentence-classification/
https://github.com/Cisco-Talos/fnc-1
https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
https://github.com/prakashpandey9/Text-Classification-Pytorch

"""

import os
import re
import torch
import torch.nn as nn
import time
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from data_prep import get_data
from dataloader import data_loader
from root import rootpath
from utils import get_tfidf_embeddings, get_bert_embeddings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LOSS_FN = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 1.4592]))
LOSS_FN = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 2.6], device=DEVICE))


# Bert
c1 = [24, 24, 32, 48, 64]
c2 = [488, 488, 248, 124, 64]
k1 = [3, 3, 3, 3]
m1 = 4
# TF IDF
c3 = [2, 2, 4, 8, 16]
k2 = [3, 3, 3, 3]
m2 = 2

def train2(model):

    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.0002)
    best_accuracy = 0

    print("Here we go...")

    for epoch in range(20):

        files = os.listdir(f"{rootpath}./embeddings_bert2/")

    return model

def train(model, x, y, epochs=10):

    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.0002)
    best_accuracy = 0

    train_inputs, val_inputs, train_labels, val_labels =\
         train_test_split(x, y, test_size=0.1, random_state=42)
    train_dataloader, val_dataloader =\
        data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=64)

    print("Start training...")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)

    for epoch_i in range(epochs):
        t0_epoch = time.time()
        total_loss = 0

        model.train()

        for _, batch in enumerate(train_dataloader):
            inputs, labels = tuple(t.to() for t in batch)
            inputs = inputs.to(device=DEVICE, dtype=torch.float)
            labels = labels.to(device=DEVICE, dtype=torch.float)
            model.zero_grad()
            logits = model(inputs)
            loss = LOSS_FN(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)

        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

    return model

def evaluate(model, val_dataloader):
    model.eval()

    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        inputs, labels = tuple(t for t in batch)
        inputs = inputs.to(device=DEVICE, dtype=torch.float)
        labels = labels.to(device=DEVICE, dtype=torch.float)

        with torch.no_grad():
            logits = model(inputs)

        loss = LOSS_FN(logits, labels)
        val_loss.append(loss.item())

        preds = format_labels(torch.reshape(torch.argmax(logits, dim=1), (-1,1)))

        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def format_labels(y):
    return torch.cat([1-y, y], dim=1)

def cnn_bert(art=True):
    # x, y, _  = get_data(use_tfidf=False)
    x, y, _ = get_bert_embeddings(art)
    y = torch.reshape(torch.tensor(y), (-1,1))
    y = format_labels(y)
    model = CNN_Bert(channel_head=c1, channels_body=c2, kernel_sizes=k1)
    model = train(model, x, y)
    path = f"{rootpath}./cnn_bert_art.pt" if art else f"{rootpath}./cnn_bert.pt"
    torch.save(model.state_dict(), path)

def cnn_tfidf(art=True):
    # x, y, _ = get_data(use_tfidf=True)
    # f = np.load(f"{rootpath}./embeddings/embeddings_tfidf.npz")
    # filenames = f.files
    # x = f[filenames[0]]
    # y = f[filenames[1]]
    # f.close()
    x, y, _ = get_tfidf_embeddings(art)
    y = torch.reshape(torch.tensor(y), (-1,1))
    y = format_labels(y)
    model = CNN_TFIDF(channels=c3, kernel_sizes=k2)
    model = train(model, x, y, epochs=15)
    path = f"{rootpath}./cnn_tfidf_art.pt" if art else f"{rootpath}./cnn_tfidf.pt"
    torch.save(model.state_dict(), path)


class CNN_TFIDF(nn.Module):
    def __init__(self, channels, kernel_sizes, kernel_size_max_pool=2):
        super(CNN_TFIDF, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_sizes[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=channels[3], out_channels=channels[4], kernel_size=kernel_sizes[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.dropout = nn.Dropout()

        self.fc = nn.Linear(256, 2)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 2, 300))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.reshape(x, (-1, 256))
        x = self.dropout(x)
        logits = self.fc(x)
        # logits = self.softmax(logits)
        return logits

class CNN_Bert(nn.Module):
    def __init__(self, channels_head, channels_body, kernel_sizes, kernel_size_max_pool=4):
        super(CNN_Bert, self).__init__()

        self.conv1_head = nn.Sequential(
            nn.Conv1d(in_channels=channels_head[0], out_channels=channels_head[1], kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv2_head = nn.Sequential(
            nn.Conv1d(in_channels=channels_head[1], out_channels=channels_head[2], kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv3_head = nn.Sequential(
            nn.Conv1d(in_channels=channels_head[2], out_channels=channels_head[3], kernel_size=kernel_sizes[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv4_head = nn.Sequential(
            nn.Conv1d(in_channels=channels_head[3], out_channels=channels_head[4], kernel_size=kernel_sizes[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv1_body = nn.Sequential(
            nn.Conv1d(in_channels=channels_body[0], out_channels=channels_body[1], kernel_size=kernel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv2_body = nn.Sequential(
            nn.Conv1d(in_channels=channels_body[1], out_channels=channels_body[2], kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv3_body = nn.Sequential(
            nn.Conv1d(in_channels=channels_body[2], out_channels=channels_body[3], kernel_size=kernel_sizes[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.conv4_body = nn.Sequential(
            nn.Conv1d(in_channels=channels_body[3], out_channels=channels_body[4], kernel_size=kernel_sizes[3]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size_max_pool)
        )

        self.dropout = nn.Dropout()

        self.fc = nn.Linear(256, 2)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_head = x[:, :24, :]
        x_body = x[:, 24:, :]
        x_head = self.conv1_head(x_head)
        x_head = self.conv2_head(x_head)
        x_head = self.conv3_head(x_head)
        x_head = self.conv4_head(x_head)
        x_body = self.conv1_body(x_body)
        x_body = self.conv2_body(x_body)
        x_body = self.conv3_body(x_body)
        x_body = self.conv4_body(x_body)
        x = torch.cat([x_head, x_body], dim=1)
        x = torch.reshape(x, (-1, 256))
        x = self.dropout(x)
        logits = self.fc(x)
        # logits = self.softmax(logits)
        return logits

if __name__ == "__main__":
    # model = CNN_TFIDF(channels=c3, kernel_sizes=k2)
    # x = torch.ones((64, 600))
    # model = CNN_Bert(channels_head=c1, channels_body=c2, kernel_sizes=k1)
    # x = torch.ones((64, 512, 768))
    # y = model(x)
    # print(y.shape)
    # cnn_tfidf(art=False)
    # cnn_bert(art=False)
    cnn_bert(art=False)