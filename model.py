import torch
import torch.nn as nn


class RNNTypeModel(nn.Module):
    def __init__(self, args, pretrained_embeddings):
        super(RNNTypeModel, self).__init__()
        self.args = args
        vocab_size, embedding_dim = pretrained_embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, args.hidden_dim, args.num_layers, dropout=args.dropout, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # x = [sentence len, batch size]
        x = torch.transpose(x, 0, 1)
        # x = [batch size, sentence len]
        embeddings = self.layernorm(self.embedding(x))
        # embeddings = [batch size, sentence len, embedding dim]
        hidden = self.init_hidden_state(embeddings.shape[0])
        out,  (hidden, _) = self.lstm(embeddings, hidden)
        # out = [batch size, sentence len, hidden dim * num directions]
        out = self.attention(out, hidden)
        out = self.dropout(self.ReLU(out))
        # out = [batch size, hidden dim * num directions]
        out = self.dropout(self.ReLU(self.fc1(out)))
        # out = [batch size, hidden dim]
        out = self.fc2(out)
        # out = [batch size, output dim]
        out = self.softmax(out)
        return out
    
    def init_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.args.num_layers * 2, batch_size, self.args.hidden_dim).zero_(),
                weight.new(self.args.num_layers * 2, batch_size, self.args.hidden_dim).zero_())
        
    def attention(self, lstm_out, hidden):
        # lstm_out = [batch size, sentence len, hidden dim * num directions]
        # hidden = [num layers * num directions, batch size, hidden dim]
        hidden = hidden[-2:].view(hidden.shape[1], -1, 1)
        # hidden = [batch size, hidden dim *num direction, 1]
        attn_weights = torch.bmm(lstm_out, hidden)
        attn_weights = torch.softmax(attn_weights, dim=1)
        # attn_weights = [batch size, sentence len, 1]
        out = torch.bmm(lstm_out.transpose(1, 2), attn_weights)
        # out = [batch size, hidden dim * num directions, 1]
        out = out.squeeze(2)
        # out = [batch size, hidden dim * num directions]
        return out
    
