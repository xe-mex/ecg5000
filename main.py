import os

import pandas
import seaborn as sns
import matplotlib as mpl
import numpy as np
from scipy.io.arff import loadarff
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch

from scipy.io.arff import loadarff
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob
import time
import copy
import shutil
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# matplotlib inline
# config InlineBackend.figure_format ='retina'

os.chdir('./ECG5000')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

mpl.rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

with open("ECG5000_TRAIN.arff") as f:
    raw_data = loadarff(f)
train = pd.DataFrame(raw_data[0])

with open('ECG5000_TEST.arff') as f:
    raw_data1 = loadarff(f)
    test = pd.DataFrame(raw_data1[0])

# print(test.head())

df = pandas.concat([train, test])
# df = train.append(test)
# print(df.head())

# print(df.shape)

df = df.sample(frac=1.0)

# print(df)

CLASS_NORMAL = 1
class_names = ['Normal', 'PVC', 'R on T', 'SP', 'UB']

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns

print(df.head())

# исследование

print(df.target.value_counts())

ax = sns.countplot(df.target)
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names)

# mpl.pyplot.show()


def plot_time_series_class(data, class_name, ax, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, linewidth=2)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.125
    )
    ax.set_title(class_name)


classes = df.target.unique()

fig, axs = plt.subplots(
    nrows=len(classes) // 3 + 1,
    ncols=3,
    sharey=True,
    figsize=(14, 8)
)

for i, cls in enumerate(classes):
    ax = axs.flat[i]
    data = df[df.target == cls] \
        .drop(labels='target', axis=1) \
        .mean(axis=0) \
        .to_numpy()
    plot_time_series_class(data, class_names[i], ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout()

mpl.pyplot.show()
