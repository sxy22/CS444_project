import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention
    query     : N x query_num x embedding_dim
    key, value: N x sentence_length x embedding_dim
    output    : N x query_num x embedding_dim
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))
             #/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_head, dropout=0.0):
        """
        h      : number of heads
        d_model: embedding size
        d_head : hidden size
        """
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_head
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_head * h), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        #  Concat
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return x


class InteractingLayer(nn.Module):
    """
    AutoInt interacting layer
    MultiHead self attention + residual connection
    """

    def __init__(self, h, d_model, d_head, features, dropout=0.0, inputproject=True, barchnorm=False):
        super(InteractingLayer, self).__init__()
        self.attention = MultiHeadedAttention(h, d_model, d_head, dropout=0.0)
        self.inputproject = inputproject
        self.barchnorm = barchnorm
        input_size = d_model
        output_size = d_head * h
        if inputproject:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.linear = nn.Linear(output_size, input_size)
        if barchnorm:
            self.bn = nn.BatchNorm1d(features)

    def forward(self, x):
        """
        Batch x Feature x embed -> Batch x Feature x embed
        """
        # self attention
        att = self.attention(x, x, x)
        # projection on input or attention output
        if self.inputproject:
            x = self.linear(x)
        else:
            att = self.linear(att)
        # residual connection
        if self.barchnorm:
            norm = self.bn(x + att)
            return F.relu(norm)
        else:
            return F.relu(x + att)


class OneHotEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(OneHotEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)

    def forward(self, x):
        return self.lut(x)


class EmbeddingLayer(nn.Module):
    """
    Batch x -1 -> Batch x Feature x d_model
    """

    def __init__(self, d_model, onehot_size=None, genre_size=10):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.genre_size = genre_size
        self.onehot_num = len(onehot_size)
        self.onehotembed = nn.ModuleList([])
        for size in onehot_size:
            self.onehotembed.append(OneHotEmbeddings(d_model, vocab=size))
        # genre embed: genre_size x d_model
        # per row is an embedding vector !
        self.genre_embed = nn.Parameter(torch.randn(genre_size, d_model))

    def forward(self, x):
        batch_size = x.shape[0]
        embed_lst = []
        for i in range(self.onehot_num):
            embed = self.onehotembed[i](x[:, i]).view(batch_size, 1, self.d_model)
            embed_lst.append(embed.clone())

        g = x[:, -self.genre_size:].type(torch.float)
        q = g.sum(dim=-1, keepdims=True)
        embed_lst.append((torch.matmul(g, self.genre_embed) / q).view(batch_size, 1, self.d_model))
        # dim=1 is the feature size dim
        # return batch x features x d_model
        return torch.cat(embed_lst, dim=1)


class BaseModelAutoInt(nn.Module):
    """
    AutoInt base model
    one Interacting layer + output layer
    Batch x Feature x d_model -> Batch x 1
    """

    def __init__(self, h, d_model, d_head, features, num_layer=1, dropout=0.0, inputproject=True, barchnorm=False):
        super(BaseModelAutoInt, self).__init__()
        self.interlayers = nn.ModuleList(
            [InteractingLayer(h, d_model, d_head, features, inputproject=inputproject, barchnorm=barchnorm)])
        if inputproject:
            hidden_size = d_head * h
        else:
            hidden_size = d_model
        # if use more than one InterLayer
        for _ in range(num_layer - 1):
            self.interlayers.append(
                InteractingLayer(h, hidden_size, d_head, features, inputproject=inputproject, barchnorm=barchnorm))
        self.output = nn.Linear(hidden_size * features, 1)

    def forward(self, x):
        # print(x.shape)
        nbatches = x.shape[0]
        # InteractingLayer
        for layer in self.interlayers:
            x = layer(x)
            # print(x.shape)
        return self.output(x.view(nbatches, -1))

class BaseModel(nn.Module):
    """
    Base model with out attention
    Batch x Feature x d_model -> Batch x 1
    """

    def __init__(self, d_model, features, dropout=0.0):
        super(BaseModel, self).__init__()
        self.output = nn.Linear(d_model * features, 1)

    def forward(self, x):
        nbatches = x.shape[0]
        return self.output(x.view(nbatches, -1))