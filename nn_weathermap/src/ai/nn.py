import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import pandas as pd
import tqdm


# それぞれの気象要素の正規化は過去期間の最大値,最小値に基づいて行う
class MinMaxScaler:
    def __init__(self):
        self.data_max = {}
        self.data_min = {}

    def set_scale(self, data: np.ndarray, var:str):
        self.data_max[var] = data.max()
        self.data_min[var] = data.min()

    def apply_scale(self, data, var:str):
        scale_data = (data - self.data_min[var]) / (self.data_max[var] - self.data_min[var])
        scale_data[scale_data < 0] = 0
        scale_data[scale_data > 1] = 1
        return scale_data

# 3D CNNモデルの定義
class Simple2DCNN(nn.Module):
    def __init__(self):
        N_lat = 31
        N_lon = 51
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            32 * N_lat * N_lon, 128
        )  # (入力次元、出力次元)このサイズは入力データのサイズに応じて調整が必要

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
