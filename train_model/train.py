import early_stopping as e
import torch
import sys
from BERT.phoBERT_sentiment import PhoBERTSentiment
from dataloader.dataloader import loader_dataset
from common_utils.load_files import load_file_csv
import torch.nn as nn


def config_train(learning_rate):
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return criterion, optimizer, scheduler


def train(model, batch_size, patience, n_epochs, trainloader, validloader, learning_rate):
    criterion, optimizer, scheduler = config_train(learning_rate)
    # To track the training loss as the model trains
    train_losses = []
    # To track the valid loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early stopping object
    early_stopping = e.EarlyStopping(patience, verbose=True)

    # Training loop
    n_total_steps = len(trainloader)
    losses = []
    evals = []
    for i in range(n_total_steps):
        model.train()
        for i, (texts, labels) in enumerate(trainloader):
            texts = texts
            labels = labels.to(torch.long)
            # Forward pass - calculate output, loss
            predicts = model(texts)
            print(predicts)
            losses = criterion(predicts, labels)
            break
        break


if __name__ == "__main__":
    model = sys.argv[1]
    if model == 'bert_sentiment_toxic':
        model = PhoBERTSentiment(4)
    df_train = load_file_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/test.csv")
    train_loader = loader_dataset(df_train)
    valid_loader = loader_dataset(df_train)
    train(model, batch_size=64, patience=7, n_epochs=5,
          trainloader=train_loader, validloader=valid_loader,
          learning_rate=3e-5)

