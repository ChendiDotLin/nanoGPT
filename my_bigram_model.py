import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class Bigram(nn.Module):  # bigram is now a subclass of nn.module
    def __init__(self, vocab_size):
        super().__init__()
        self.model_ = nn.Embedding(vocab_size, vocab_size)
        # it is a map from an idx in the range of [0, vocab_size) to a row with logits

    def forward(self, idx, gt):
        # idx ~[B, N]
        logits = self.model_(idx)
        # [B, N, C], C is vocab_size, basically the logits to show the probability
        # print("idx shape: ", idx.shape)
        # print("logits shape: ", logits.shape)
        # print("gt shape: ", gt.shape)
        B, N, C = logits.shape
        if gt is None:
            return logits, None
        loss = F.cross_entropy(logits.view((B * N, -1)), gt.view((B * N)))
        return logits, loss

    def generate(self, context, new_token_size):
        # context ~[B,N]
        for _ in range(new_token_size):
            # # this model only uses the last token
            # idx = context[:, -1]
            # print(idx.shape)
            logits, _ = self.forward(context, None)
            # [B, N, C]
            # print(logits.shape)
            probability = torch.softmax(logits[:, -1, :], dim=-1)  # [B,C]
            # print(probability.shape)
            new_token = torch.argmax(probability, dim=-1, keepdim=True)
            print(context.shape)
            print(new_token.shape)
            context = torch.cat((context, new_token), dim=1)
        return context[:, -new_token_size:]


enc = tiktoken.get_encoding("gpt2")
# lets read data/shakespeare/train.bin and see how that goes
train_data_path = "data/shakespeare/train.bin"
train_data = np.fromfile(train_data_path, dtype=int)
# print(len(train_data))  # the total length of shakespeare tokens

txt = enc.decode(train_data[:100])
vocab_size = enc.n_vocab
# print(vocab_size)
block_size = 16  # we use 16 tokens as a block
# x = torch.from_numpy(train_data[:block_size])
# y = torch.from_numpy(train_data[1 : block_size + 1])  # y is shifted 1 away from x
# for t in range(block_size):
#     context = x[: t + 1]  # grows from 1 to block_size
#     target = y[t]  # one last position
#     print("context: ", context, " target: ", target)

block_size = 8
batch_size = 32


# create batch given random int
def get_batch(data):
    # data is a long numpy array
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size]) for i in idx])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1]) for i in idx])
    return x, y


# x, y = get_batch(train_data)
# print(x, y)
m = Bigram(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for _ in range(10000):
    x, y = get_batch(train_data)
    # _, loss = m(x, y)
    # print(loss)
    # output = m.generate(x, 1)
    # print("context: ", x)
    # print("output: ", output)
    _, loss = m.forward(x, y)
    optimizer.zero_grad()
    print("loss: ", loss)
    loss.backward()
    optimizer.step()
