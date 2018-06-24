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


class VanillaRNN(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=50, output_size=len(all_categories)):
        super(VanillaRNN, self).__init__()
        self.in_size, self.hid_size, self.out_size = input_size, hidden_size, output_size
        self.hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, h):
        stacked = torch.cat((x, h), dim=1)
        h = F.sigmoid(self.hidden(stacked))
        o = F.log_softmax(self.output(stacked), dim=1)
        return o, h

    def init_hidden(self):
        return torch.zeros(1, self.hid_size)

    def forward_all_sequence(self, x):
        h = self.init_hidden()
        for i in range(x.size()[0]):
            o, h = self(x[i], h)
        return o


def train(net, n_iterations, criterion, optimizer, print_every=5000):
    assert print_every > 0
    net.train()
    print "+-----------+-----------+----------+----------------+"
    print "| Iteration | Avg. Loss | Accuracy | Time (seconds) |"
    print "+-----------+-----------+----------+----------------+"
    total_loss = good = 0.0
    start = time()
    for i in xrange(n_iterations):
        _, _, y, x = randomTrainingExample()
        net.zero_grad()
        out = net.forward_all_sequence(x)
        loss = criterion(out, y)
        good += (torch.argmax(out) == y).item()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print
        if i % print_every == print_every - 1:
            print "| {:^9} | {:^ 8.6f} | {:^07.3f}% | {:^14.5f} |".format(
                i+1, total_loss / print_every, 100.0 * good / print_every, time() - start)
            total_loss = good = 0.0
            start = time()
    print "+-----------+-----------+----------+----------------+"


def test(net, criterion, test_set):
    net.eval()
    preds = []
    total_loss = good = 0.0
    for x, y in test_set:
        out = net.forward_all_sequence(x)
        prediction = torch.argmax(out).item()
        preds.append(prediction)
        good += prediction == y.item()
        total_loss += criterion(out, y).sum()
    return total_loss / len(test_set), good / len(test_set), preds


def plot_confusion_matrix(cm, title):
    f = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(all_categories))
    plt.xticks(tick_marks, all_categories, rotation=45)
    plt.yticks(tick_marks, all_categories)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    f.savefig("cm_vanilla.png")


def main():
    # parameters
    n_iterations = 10 ** 5
    lr = 0.005

    net = VanillaRNN()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    train(net, n_iterations, criterion, optimizer)

    test_set = []
    true_labels = []
    for c, category in enumerate(all_categories):
        for line in file("data/test/{}.txt".format(category)):
            if line == "\n":
                continue
            test_set.append((lineToTensor(line[:-1]), torch.tensor([c], dtype=torch.long)))
            true_labels.append(c)
    loss, accuracy, predictions = test(net, criterion, test_set)
    print "\nLoss on test set is {} and Accuracy is {}%".format(loss, accuracy * 100.0)
    cm = confusion_matrix(true_labels, predictions, range(len(all_categories)))
    print np.array(cm)
    plot_confusion_matrix(cm, "Vanilla RNN Confusion Matrix")

if __name__ == '__main__':
    main()