import numpy as np

import train_model.early_stopping as e
import torch
import sys
from BERT.phoBERT_sentiment import PhoBERTSentiment
from dataloader.dataloader import loader_dataset
from common_utils.load_files import load_file_csv
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import time


def config_train(learning_rate):
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return criterion, optimizer, scheduler


def train_batch(dataloader, id_e):
    model.train()
    # Track accuracy and count
    total_acc, total_count = 0, 0

    # Track train_loss
    train_losses = []
    total_iter = len(dataloader)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    for idx, (texts, labels) in pbar:
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

        total_acc += (results == labels).sum().item()
        total_count += labels.size(0)
        train_losses.append(losses.detach().numpy())

        if idx % 25 == 0:
            msg = f'[Epoch {id_e}|{n_epochs}] Iter {idx}|{total_iter} | ' \
                  f'Loss: {np.average(train_losses):.5f} | Accuracy: {(total_acc/total_count):.5f}'
            print(msg)
    return np.average(train_losses)


def valid_batch(dataloader):
    total_acc, total_count = 0, 0
    val_losses = []

    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    for i, (texts, labels) in pbar:
        with torch.no_grad():
            texts = texts.to(device)
            labels = labels.to(torch.long).to(device)

            # Forward pass
            predicts = model(texts)
            losses = criterion(predicts, labels)
            total_acc += (predicts.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
            val_losses.append(losses.detach().numpy())
    valid_accuracy = total_acc/total_count
    print(f'Accuracy of epoch {valid_accuracy}')
    return np.average(val_losses), valid_accuracy


def train(model, batch_size, patience, n_epochs, trainloader, validloader, learning_rate, num_valid):
    criterion, optimizer, scheduler = config_train(learning_rate)
    # To track the training loss as the model trains
    train_losses = []
    # To track the valid loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # best accuracy
    best_acc = 0

    # initialize the early stopping object


    # Training loop
    n_total_steps = len(trainloader)
    losses = []
    evals = []
    for epoch in range(n_epochs):
        num_true_value = 0
        model.train()
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
        for t, (texts, labels) in pbar:
            texts = texts
            labels = labels.to(torch.long)

            # Forward pass - calculate output, loss
            predicts = model(texts)
            loss = criterion(predicts, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recording training loss
            train_losses.append(loss)

        model.eval()
        with torch.no_grad():
            for i, (texts, labels) in enumerate(validloader):
                labels = labels.to(torch.long)
                # Forward pass
                predicts = model(texts)
                results = torch.argmax(predicts, dim=1).numpy()

                valid_loss = criterion(predicts, labels)
                valid_losses.append(valid_loss)

                conf_matrix = confusion_matrix(labels.numpy(), results)
                for index in range(conf_matrix.shape[0]):
                    num_true_value += conf_matrix[index][index]

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f} '
                     )
        print(print_msg)

        valid_acc = num_true_value / num_valid
        if valid_acc > best_acc:
            best_acc = valid_acc
            print(f'Saving model .....')
            torch.save(model.state_dict(),
                       "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/models/sentiment/best_acc.pth")

        # clear loss to track in the next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint
    model.load_state_dict(torch.load('/Users/tieuanhnguyen/PycharmProjects/FinalThesis/models/sentiment/best_acc.pth'))
    return model, avg_train_losses, avg_valid_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sys.argv[1]
    if model == 'bert_sentiment_toxic':
        model = PhoBERTSentiment(4)
    df_train = load_file_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/train_upsampling.csv")
    df_valid = load_file_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/dev.csv")
    train_loader = loader_dataset(df_train)
    valid_loader = loader_dataset(df_train)

    # Config criterion, optimizer, scheduler
    criterion, optimizer, scheduler = config_train(learning_rate=1e-5)

    num_valid = len(df_valid)
    n_epochs = 5
    val_loss, train_loss = 0, 0
    acc_total = None

    for id in range(n_epochs):
        epoch_start_time = time.time()

        early_stopping = e.EarlyStopping(patience=5, verbose=True)

        train_loss = train_batch(train_loader, id)
        valid_loss, valid_acc = valid_batch(valid_loader)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
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
            torch.save(model.state_dict(), "models/sentiment/best_acc_sent_toxic.pth")

        epoch_end_time = time.time()
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                id, time.time() - epoch_start_time, valid_acc)
        )