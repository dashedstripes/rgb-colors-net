import torch
from torch import nn

labels = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

m = nn.Softmax(dim=0)

def normalize_rgb(rgb):
  # normalize the tensor values so that dark colors are max 0.5, and light colors are max 255
  v = torch.tensor([rgb[0], rgb[1], rgb[2]], dtype=torch.float32)
  v /= 255
  return v

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

def run(rgb):
  # normalize the inputs
  v = normalize_rgb(rgb)

  # run the data
  model = torch.load("models/rgb_nn.pth")
  model.eval()

  logits = model(v)
  pred = torch.argmax(m(logits))

  return print(labels[pred])

run([109, 168, 181])
