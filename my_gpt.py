import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


enc = tiktoken.get_encoding("gpt2")
# lets read data/shakespeare/train.bin and see how that goes
train_data_path = "data/shakespeare/train.bin"
train_data = np.fromfile(train_data_path, dtype=int)
# print(len(train_data))  # the total length of shakespeare tokens
# print("training data: ", enc.decode(train_data))

txt = enc.decode(train_data[:100])
vocab_size = enc.n_vocab
# print(vocab_size)
# x = torch.from_numpy(train_data[:block_size])
# y = torch.from_numpy(train_data[1 : block_size + 1])  # y is shifted 1 away from x
# for t in range(block_size):
#     context = x[: t + 1]  # grows from 1 to block_size
#     target = y[t]  # one last position
#     print("context: ", context, " target: ", target)

block_size = 8
batch_size = 32
hidden_layer_dim = 32
# head_size = 16
learning_rate = 1e-3
num_iter = 1000
head_num = 4
num_blocks = 3
eps = 1e-6

context = train_data[:block_size]
print("context: ", enc.decode(context))

context = torch.from_numpy(context).view(1, -1)


class layernorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (torch.sqrt(var) + eps) + self.beta


class ffd(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, input_size),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.linear(x)


class selfAttention(nn.Module):
    """one head attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(hidden_layer_dim, head_size, bias=False)
        self.value = nn.Linear(hidden_layer_dim, head_size, bias=False)
        self.query = nn.Linear(hidden_layer_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, d = x.shape
        v = self.key(x)
        k = self.value(x)
        q = self.query(x)
        correlation = q @ k.transpose(-1, -2)  # B, T, T
        correlation *= d**0.5  # *sqrt(d)
        weight = torch.masked_fill(correlation, self.tril == 0, float("-inf"))
        weight = torch.softmax(weight, dim=-1)
        return weight @ v  # B,T,d


class MultiHeadAttention(nn.Module):
    """multi head attention by concat all the results"""

    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([selfAttention(head_size) for _ in range(head_num)])

    def forward(self, x):
        # B,t,d = x.shape
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # every head is only taking d//head_num now


class Block(nn.Module):
    """each block is a multi head with a ffd"""

    def __init__(self):
        super().__init__()
        self.ln1 = layernorm(hidden_layer_dim)
        self.ln2 = layernorm(hidden_layer_dim)
        self.self_attention = MultiHeadAttention(head_num, hidden_layer_dim // head_num)
        self.ffd = ffd(hidden_layer_dim, hidden_layer_dim * 4)

    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))  # residual
        x = x + self.ffd(self.ln2(x))
        return x


class nanoGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, hidden_layer_dim)
        self.pos_embedding = nn.Embedding(block_size, hidden_layer_dim)
        # * unpack the list so the squential will see the input as (block, block, block)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.ln = layernorm(hidden_layer_dim)
        # this generate logits
        self.linear = nn.Linear(hidden_layer_dim, vocab_size)

    def forward(self, idx, gt=None):
        B, T = idx.shape
        vocab_embedding = self.vocab_embedding(idx)
        pos = torch.arange(T)
        pos_embedding = self.pos_embedding(pos)  # T,C
        embedding = vocab_embedding + pos_embedding  # B,T,C
        embedding = self.blocks(embedding)
        embedding = self.ln(embedding)
        logits = self.linear(embedding)  # B,T,vocab_size
        if gt is not None:
            loss = nn.functional.cross_entropy(logits.view(B * T, -1), gt.view(B * T))
            return logits, loss
        return logits, None

    def generate(self, context, new_token_size):
        # context ~[B,N]
        for _ in range(new_token_size):
            # # this model only uses the last token
            # idx = context[:, -1]
            # print(idx.shape)
            idx = context[:, -block_size:]
            logits, _ = self.forward(idx, None)
            # [B, N, C]
            # print(logits.shape)
            probability = torch.softmax(logits[:, -1, :], dim=-1)  # [B,C]
            # print(probability.shape)
            new_token = torch.multinomial(probability, num_samples=1)
            # print(context.shape)
            # print(new_token.shape)
            context = torch.cat((context, new_token), dim=1)
        return context[:, -new_token_size:]


def get_batch(data):
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x, y


train_data = torch.from_numpy(train_data)

m = nanoGPT()
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for _ in range(num_iter):
    x, y = get_batch(train_data)
    _, loss = m(x, y)
    optimizer.zero_grad()
    loss.backward()
    print("loss: ", loss)
    optimizer.step()

# this is the trained model
m.eval()

generated = m.generate(context, 100)
generated_word = enc.decode(generated[0].tolist())
print("generated", generated_word)
