# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""CTEFNet base class"""
import mindspore.nn as nn
from mindspore import ops


class CTEFNet(nn.Cell):
    
    def __init__(self,
                 cov_hidden_channels=60,
                 cov_out_channels=15,
                 heads=3,
                 num_layer=4,
                 feedforward_dims=256,
                 dropout=0.1,
                 obs_time=12,
                 pred_time=24
                 ):
        super().__init__()
        self.cov_hidden_channels = cov_hidden_channels
        self.cov_out_channels = cov_out_channels
        self.heads = heads
        self.num_layer = num_layer
        self.feedforward_dims = feedforward_dims
        self.dropout = dropout

        self.obs_time = obs_time
        self.pred_time = pred_time

        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=self.cov_hidden_channels, kernel_size=(4, 8), pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=self.cov_hidden_channels, out_channels=self.cov_hidden_channels, kernel_size=(2, 4),
                      pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=self.cov_hidden_channels, out_channels=self.cov_out_channels, kernel_size=(2, 4),
                      pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((3, 6)),
            nn.Flatten()])

        encoderlayer = nn.TransformerEncoderLayer(3 * 6 * self.cov_out_channels, self.heads,
                                                  dim_feedforward=feedforward_dims, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoderlayer, num_layers=self.num_layer)
        self.res = nn.Dense(18 * self.cov_out_channels * self.obs_time, 18 * self.cov_out_channels * self.obs_time)
        self.head = nn.Dense(18 * self.cov_out_channels * self.obs_time, self.obs_time + self.pred_time)

    def construct(self, x):
        fea = ops.unsqueeze(self.conv(x[:, 0, :, :, :]), dim=1)
        for c in range(1, self.obs_time):
            fea = ops.cat([fea, ops.unsqueeze(self.conv(x[:, c, :, :, :]), dim=1)], axis=1)
        out = self.encoder(fea)

        fea = ops.flatten(fea)
        out = ops.flatten(out)
        fea = self.res(fea)
        out = fea + out
        out = self.head(out)
        return out
