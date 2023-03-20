import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.target_transform = target_transform
        self.transform = transform
        self.X = X
        self.y = y + 2

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X, y = self.X[idx, :], self.y[idx]

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.transform(y)

        return X, y


def draw(train_losses, test_losses, train_f1_scores, test_f1_scores):
    clear_output()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(test_losses, label='test')
    plt.plot(train_losses, label='train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(test_f1_scores, label='test')
    plt.plot(train_f1_scores, label='train')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.grid()

    plt.show()


def train(model, optimizer, epochs, train_loader, test_loader):
    criterion = nn.NLLLoss()
    train_losses = []
    test_losses = []
    train_f1_scores = []
    test_f1_scores = []

    for i in tqdm(range(epochs)):
        train_loss_sum = 0
        test_loss_sum = 0
        train_f1_sum = 0
        test_f1_sum = 0

        # train
        model.train()
        for X, y in iter(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            train_loss_sum += loss.item()
            train_f1_sum += f1_score(y.cpu(), torch.argmax(y_pred, dim=1).cpu(), average='macro')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # plus pre step here
        train_losses.append(train_loss_sum / len(train_loader))
        train_f1_scores.append(train_f1_sum / len(train_loader))

        # test
        model.eval()
        with torch.no_grad():
            for X, y in iter(test_loader):
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                test_loss_sum += loss.item()
                test_f1_sum += f1_score(y.cpu(), torch.argmax(y_pred, dim=1).cpu(), average='macro')
            test_losses.append(test_loss_sum / len(test_loader))
            test_f1_scores.append(test_f1_sum / len(test_loader))

        draw(train_losses, test_losses, train_f1_scores, test_f1_scores)

    return train_losses, test_losses, train_f1_scores, test_f1_scores
