"""
General utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or tensor int, shape=(N, )
    Returns:
        masks: bool tensor, shape=(N, L), padded=False
    """
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.long)
    if max_len is None:
        max_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    if max_len == 0:
        return torch.zeros((seq_lens.numel(), 0), dtype=torch.bool)
    return torch.arange(max_len).unsqueeze(0) < seq_lens.unsqueeze(1)

