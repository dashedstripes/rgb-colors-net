import random
import csv
from colorsys import hls_to_rgb

hues = {
  "red": {
    "low": 350,
    "high": 359
  },
  "orange": {
    "low": 15,
    "high": 45
  },
  "yellow": {
    "low": 50,
    "high": 65
  },
  "green": {
    "low": 70,
    "high": 160
  },
  "blue": {
    "low": 170,
    "high": 250
  },
  "purple": {
    "low": 260,
    "high": 280
  },
  "pink": {
    "low": 290,
    "high": 340
  }
}

hue_choices = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

def generate_hue(color="red"):
  return color

def generate_color(color="red"):
  h=random.randrange(int(hues[color]['low']), int(hues[color]['high']))
  s=random.randrange(15, 100)
  l=random.randrange(8, 93)

  # print("debug hue", h, s, l)

  r, g, b = hls_to_rgb(h/360, l/100, s/100)
  return [round(r*255.0), round(g*255.0), round(b*255.0)]

def generate_csv(num_of_rows=50000, filename='rgb_train.csv'):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['red', 'green', 'blue', 'label'])

    for _ in range (num_of_rows):
      label = random.choice(hue_choices)
      color = generate_color(label)
      writer.writerow([color[0], color[1], color[2], label])

generate_csv(num_of_rows=50000, filename='rgb_train.csv')
generate_csv(num_of_rows=50000, filename='rgb_test.csv')