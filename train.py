import numpy as np

import train_model.early_stopping as e
import torch
import sys
from BERT.phoBERT_sentiment import PhoBERTSentiment, PhoBertBiLSTM, PhoBertLSTM, PhoBertCNN
from dataloader.dataloader import loader_dataset
from common_utils.load_files import load_file_csv
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm
import time
from dataloader.textloader import load_preprocessing_data
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd


def compute_precision(labels, predicts):
    precision = precision_score(labels, predicts, average="macro")
    return precision


def compute_recall(labels, predicts):
    recall = recall_score(labels, predicts, average="macro")
    return recall


def compute_f1(labels, predicts):
    precision = compute_precision(labels, predicts)
    recall = compute_recall(labels, predicts)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_acc_labels(labels, predicts):
    matrix = confusion_matrix(labels, predicts)
    n_labels = len(matrix)
    acc_labels = list(0. for i in range(n_labels))
    total_labels = list(0. for i in range(n_labels))
    for i in range(n_labels):
        total_labels[i] = matrix[i].sum()
        acc_labels[i] = matrix[i][i] / total_labels[i]
        print(f'Accuracy label {i}: {matrix[i][i]} | {total_labels[i]}')
    return acc_labels


def config_train(model, learning_rate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torch.tensor([2, 8], dtype=torch.float32).to(device)
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return criterion, optimizer, scheduler


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
  """
    # for params in m.lstm.parameters():
    #     params.reset_parameters()
    #
    # for params in m.head_model.parameters():
    #     params.reset_parameters()
    for layer in m.children():
        # print(layer)
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def train_batch(model, device, optimizer, criterion, dataloader, id_e):
    model.train()
    model.to(device)
    # Track accuracy and count
    total_acc, total_count = 0, 0
    total_acc_0_1 = [0, 0]
    total_count_0_1 = [0, 0]

    # Track train_loss
    train_losses = []
    total_iter = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=total_iter, leave=True)
    for idx, (texts, labels) in enumerate(dataloader):
        texts = texts.to(device)
        labels = labels.to(torch.long).to(device)
        optimizer.zero_grad()

        # Forward pass  (calculate predicts, losses)
        predicts = model(texts)
        results = torch.argmax(predicts, dim=1).to(torch.long)
        losses = criterion(predicts, labels)

        # Backward pass
        losses.backward()
        optimizer.step()

        matrix = confusion_matrix(labels.detach().cpu().numpy(), predicts.argmax(dim=1).detach().cpu().numpy())
        try:
            total_acc_0_1[0] += matrix[0][0]
            total_acc_0_1[1] += matrix[1][1]
        except Exception as e:
            print(e)
        total_count_0_1[0] += (labels == 0).sum().item()
        total_count_0_1[1] += (labels == 1).sum().item()

        total_acc += (results == labels).sum().item()
        total_count += labels.size(0)
        train_losses.append(losses.detach().cpu().numpy())

        if idx % 105 == 0:
            msg = f'[Epoch {id_e}|{params["n_epochs"]}] Iter {idx}|{total_iter} | ' \
                  f'Loss: {np.average(train_losses):.5f} | Accuracy: {(total_acc / total_count):.5f} | ' \
                  f'Accuracy 0: {total_acc_0_1[0] / total_count_0_1[0]} ({total_acc_0_1[0]}|{total_count_0_1[0]}) | ' \
                  f'Accuracy 1: {total_acc_0_1[1] / total_count_0_1[1]} ({total_acc_0_1[1]}|{total_count_0_1[1]})'

            print(msg)
    return model, np.average(train_losses)


def valid_batch(model, device, criterion, dataloader):
    total_acc, total_count = 0, 0
    val_losses = []
    true_values = []
    preds = []

    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    for i, (texts, labels) in pbar:
        with torch.no_grad():
            texts = texts.to(device)
            labels = labels.to(torch.long).to(device)

            # Forward pass
            predicts = model(texts)
            losses = criterion(predicts, labels)
            # Concat labels and predict
            true_values = true_values + labels.detach().cpu().numpy().tolist()
            preds = preds + predicts.argmax(dim=1).detach().cpu().numpy().tolist()

            total_acc += (predicts.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
            val_losses.append(losses.detach().cpu().numpy())
    valid_accuracy = total_acc / total_count
    val_acc_labels = compute_acc_labels(true_values, preds)
    recall = compute_recall(true_values, preds)
    print(f'Accuracy valid of epoch {valid_accuracy}')
    print(f'Precision valid: {compute_precision(true_values, preds)}')
    print(f'Recall valid: {compute_recall(true_values, preds)}')
    print(f'F1-score: {compute_f1(true_values, preds)}')
    for i in range(len(val_acc_labels)):
        print(f'Accuracy valid of label {i}: {val_acc_labels[i]}')
    return model, np.average(val_losses), valid_accuracy, recall


def test_valid(model, df_test):
    results_correct = 0
    count_0_1 = [0, 0]
    result_correct_0_1 = [0, 0]
    model.eval()
    X_test, y_test = load_preprocessing_data(df_test)
    labels = np.array(y_test)
    count_0_1[0] = (labels == 0).sum()
    count_0_1[1] = (labels == 1).sum()
    count_text = len(labels)
    for i in range(len(X_test)):
        text = df_test['free_text'][i]
        x = torch.tensor(X_test[i].reshape((1, -1)), dtype=torch.long).to(device)
        predict = model(x).argmax(dim=1)
        if predict == y_test[i] and y_test[i] == 1:
            results_correct += 1
            result_correct_0_1[1] += 1
        elif predict == y_test[i] and y_test[i] == 0:
            results_correct += 1
            result_correct_0_1[0] += 1
        count_text += 1
    valid_0_accuracy = result_correct_0_1[0] / count_0_1[0]
    valid_1_accuracy = result_correct_0_1[1] / count_0_1[1]
    print(f'Accuracy valid of epoch in value 0: {valid_0_accuracy} [{result_correct_0_1[0]}|{count_0_1[0]}]')
    print(f'Accuracy valid of epoch in value 1: {valid_1_accuracy} [{result_correct_0_1[1]}|{count_0_1[1]}]')
    return results_correct


def train(model_name, df_train, df_valid, df_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    # Set fixed random number seed
    torch.manual_seed(42)

    # For fold results
    results = {}

    train_dataset = loader_dataset(df_train)
    valid_dataset = loader_dataset(df_valid)
    test_dataset = loader_dataset(df_test)
    dataset = ConcatDataset([train_dataset, valid_dataset])

    # Define K-Fold cross validator
    kfold = KFold(n_splits=params['k_folds'], shuffle=True)

    # Start Printing
    print(f'---------------------------------------------------')
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=train_subsampler)
        valid_loader = DataLoader(dataset, batch_size=params["batch_size"], sampler=test_subsampler)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

        # Init Neural Network
        if model_name == "BertBiLSTM":
            model = PhoBertBiLSTM(2)
        elif model_name == "BertSent":
            model = PhoBERTSentiment(4)
        elif model_name == "BertLSTM":
            model = PhoBertLSTM(2)
        elif model_name == "BertCNN":
            model = PhoBertCNN(2)

        # Print model
        print(model)
        model.apply(reset_weights)

        # Init optimizer, loss function, scheduler
        criterion, optimizer, scheduler = config_train(model, learning_rate=params['learning_rate'])

        # Freeze param phoBERT
        for param in model.base_model.parameters():
            param.requires_grad = False

        acc_total = None

        # Run the training loop for defined num epochs
        early_stopping = e.EarlyStopping(patience=params["patience"], verbose=True)
        for epoch in range(params["n_epochs"]):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            epoch_start_time = time.time()

            # Iterate over the DataLoader for training data
            model, train_loss = train_batch(model, device, optimizer, criterion, train_loader, epoch)
            model, valid_loss, valid_acc, recall_valid = valid_batch(model, device, criterion, valid_loader)

            print(f'------Test----------')
            model, test_loss, test_acc, recall_valid = valid_batch(model, device, criterion, test_loader)
            print(f'--------End Test---------')
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if acc_total is not None and acc_total > valid_acc:
                scheduler.step()
            else:
                acc_total = valid_acc
                print(f"Saving best model: valid_loss: {valid_loss}")
                print(f"Best metric: accuracy: {acc_total}")
                torch.save(model.state_dict(), "models/sentiment/best_acc_sent_toxic_ver4.pth")

            print(
                f"| end of epoch {epoch} | time: {time.time() - epoch_start_time:5.2f}s | valid accuracy {valid_acc:8.3f} "
            )


if __name__ == "__main__":
    params = {
        "batch_size": 64,
        "learning_rate": 1e-5,
        "k_folds": 5,
        "n_epochs": 20,
        "patience": 5,

    }

    df_train = load_file_csv("crawler/test_text_crawler/train_bitoxic_update.csv")
    df_valid = load_file_csv("synonyms_upsampling/vihsd/dev.csv")
    df_test = load_file_csv("synonyms_upsampling/vihsd/test.csv")
    train("BertCNN", df_train, df_valid, df_test)