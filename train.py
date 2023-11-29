import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText 
import argparse
from model import RNNTypeModel
import os
import matplotlib.pyplot as plt

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up fields
    TEXT = data.Field()
    LABEL = data.Field(sequential=False, dtype=torch.long)
    
    # make splits for data
    # DO NOT MODIFY: fine_grained=True, train_subtrees=False
    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False)
    
    # build the vocabulary
    if args.embed_type =='vector':
        TEXT.build_vocab(train, vectors=Vectors(name='vector.txt', cache='./data'))
    elif args.embed_type =='glove':
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    else:
        TEXT.build_vocab(train, vectors=FastText(language='en'))

    LABEL.build_vocab(train)
    
    # make iterator for splits
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=args.batch_size, shuffle=True)
    
    # Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
    pretrained_embeddings = TEXT.vocab.vectors

    # build the model
    model = RNNTypeModel(args, pretrained_embeddings)
    model.to(device)
    # train the model
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = 0.0
    early_stop = 0

    os.makedirs(args.save_dir, exist_ok=True) 

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(1, args.epochs+1):
        total_loss = 0
        total_correct = 0
        total_num = 0
        for (batch_num, batch) in enumerate(train_iter):
            model.train()
            inputs, labels = batch.text, batch.label - 1
            if args.output_dim == 3:
                labels = torch.where(labels < 2, 0, labels)
                labels = torch.where(labels == 2, 1, labels)
                labels = torch.where(labels > 2, 2, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            nll_loss = criterion(outputs, labels)
            loss = nll_loss.mean()
            _, predict = torch.max(outputs, dim=1)
            correct = (predict == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += nll_loss.sum().item()
            total_correct += correct
            total_num += len(labels)

            if batch_num % args.log_interval == 0:
                # print log
                loss = total_loss / total_num
                acc = total_correct / total_num
                print('Epoch: {}, Batch num: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, batch_num, loss, acc))
        loss = total_loss / total_num
        acc = total_correct / total_num
        train_loss.append(loss)
        train_acc.append(acc)
        print('--- Epoch {} Train Result ---'.format(epoch))
        print('Train loss: {:.4f}, Train acc: {:.4f}'.format(loss, acc))

        loss, acc = evaluate(args, model, val_iter, criterion)
        val_loss.append(loss)
        val_acc.append(acc)
        print('--- Epoch {} Val Result ---'.format(epoch))
        print('Val loss: {:.4f}, Val acc: {:.4f}'.format(loss, acc))

        # plot
        x = range(epoch)
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(121)
        ax1.plot(x, train_loss, label='training loss')
        ax1.plot(x, val_loss, label='valid loss')
        ax1.set_ylabel("loss")
        ax1.set_xlabel("epoch")
        plt.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(x, train_acc, label='training accuracy')
        ax2.plot(x, val_acc, label='valid accuracy')
        ax2.set_ylabel("accuracy")
        ax2.set_xlabel("epoch")

        plt.legend()
        plt.savefig(args.save_dir + "/loss_and_acc.png")
        plt.close()

        # early stop
        if acc > best_val_acc:
            best_val_acc = acc
            early_stop = 0
            torch.save(model, os.path.join(args.save_dir, 'model_best.pt'))
        else:
            early_stop += 1
        if early_stop == args.early_stop:
            print("Early stop at {} epoch!".format(epoch))
            break
    # test the best model
    model = torch.load(args.save_dir + "/model_best.pt")
    val_loss, val_acc = evaluate(args, model, val_iter, criterion)
    test_loss, test_acc = evaluate(args, model, test_iter, criterion)
    print('--- Test Result ---'.format(epoch))
    print('Val loss: {:.4f}, Val acc: {:.4f}, Test loss {:.4f}, Test acc {:.4f}'.format(val_loss, val_acc, test_loss, test_acc))


    
def evaluate(args, model, test_iter, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_num = 0
        for batch in test_iter:
            inputs, labels = batch.text, batch.label - 1
            if args.output_dim == 3:
                labels = torch.where(labels < 2, 0, labels)
                labels = torch.where(labels == 2, 1, labels)
                labels = torch.where(labels > 2, 2, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            nll_loss = criterion(outputs, labels)
            _, predict = torch.max(outputs, dim=1)
            correct = (predict == labels).sum().item()

            total_loss += nll_loss.sum().item()
            total_correct += correct
            total_num += len(labels)
        
        loss = total_loss / total_num
        acc = total_correct / total_num
    return loss, acc

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch RNN Type Example')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for train [default: 100]')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--embedding-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
    parser.add_argument('--hidden-dim', type=int, default=300, help='number of hidden dimension [default: 300]')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers [default: 2]')
    parser.add_argument('--output-dim', type=int, default=5, help='number of output dimension [default: 5]')
    parser.add_argument('--log-interval', type=int, default=10, help='how many steps to wait before logging training status [default: 10]')
    parser.add_argument('--save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('--early-stop', type=int, default=30, help='iteration numbers to stop without performance increasing')
    parser.add_argument('--embed-type', type=str, choices=['vector','glove','fasttext'], default='vector', help='embedding type: vector, glove, fasttext')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    train_model(args)




   
