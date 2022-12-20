import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

num_of_epochs = 20
learning_rate = 1e-2
batch_size = 64

labels = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

def normalize_rgb(rgb):
  v = torch.tensor([rgb[0], rgb[1], rgb[2]], dtype=torch.float32)
  v /= 255
  return v

class RGBDataset(Dataset):
  def __init__(self, data_file):
    self.data = pd.read_csv(data_file)
    self.one_hot = nn.functional.one_hot(torch.arange(0, 7), 7)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    raw_value = self.data.iloc[idx, 0:3]
    raw_label = self.data.iloc[idx, 3]

    # one hot encode label here
    v = normalize_rgb(raw_value)
    label = self.one_hot[labels.index(raw_label)]

    # https://github.com/pytorch/pytorch/issues/42188
    return v, torch.as_tensor(label, dtype=torch.float32)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 7),
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

training_data = RGBDataset("data/rgb_train.csv")
test_data = RGBDataset("data/rgb_test.csv")

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# train_features, train_labels = next(iter(train_dataloader))
# print(train_features, train_labels)

model = NeuralNetwork().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
m = nn.Softmax(dim=1)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction and loss
    logits = model(X)
    pred = m(logits)

    # https://discuss.pytorch.org/t/cross-entropy-loss-is-not-decreasing/43814
    loss = loss_fn(logits, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # I am leaving this here because it's useful to know that the weights are updating, it helped me a lot during debugging
    # for name, param in model.named_parameters():
    #   print(name, param.grad)

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      logits = model(X)
      pred = m(logits)

      test_loss += loss_fn(pred, y).item()

      index_of_correct = torch.argmax(pred, dim=1)
      correct += torch.sum(index_of_correct == torch.argmax(y, dim=1)).item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def run(save_model=False):
  epochs = num_of_epochs
  for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
  print("Done!")

  if save_model:
    print("Saving Model")
    torch.save(model, 'models/rgb_nn.pth')
    print('Saved!')

run(save_model=False)