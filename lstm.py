from io import open
import glob
import torch
import random
import unicodedata
import string
from torch.nn import functional as F
from torch import nn
from time import time
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

"############## Utils ##############"


def findFiles(path):
    return glob.glob(path)


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/train/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


"############## Training ##############"


class LSTM(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=50, output_size=len(all_categories)):
        super(LSTM, self).__init__()
        self.in_size, self.hid_size, self.out_size = input_size, hidden_size, output_size
        self.input = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        stacked = torch.cat((x, h), dim=1)
        i, f, o = map(F.sigmoid, [self.input(stacked), self.forget(stacked), self.output_gate(stacked)])
        g = F.tanh(self.gate(stacked))
        c = f * c + i * g
        h = o * F.tanh(c)
        o = F.log_softmax(self.output(h), dim=1)
        return o, h, c

    def init_hidden_and_cell(self):
        return torch.zeros(1, self.hid_size), torch.zeros(1, self.hid_size)

    def forward_all_sequence(self, x):
        h, c = self.init_hidden_and_cell()
        for i in range(x.size()[0]):
            o, h, c = self(x[i], h, c)
        return o