# Copyright 2020 LMNT, Inc. All Rights Reserved.
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
# ==============================================================================

import os
import torch
import torchaudio

from functools import partial
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from src.util.acoustic_utils import clean_noisy_subsample


def get_clean_name(noisy_name):
    file_id = noisy_name.split('_')[-1]
    test_path = '/'.join(noisy_name.split('/')[:-2])
    return os.path.join(test_path, 'clean', 'clean_fileid_' + file_id)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, sr, sub_sample_length):
        super().__init__()
        sub_sample_samples = int(sr*sub_sample_length)

        self.sub_sample_length = sub_sample_length
        self.filenames = glob(f'{path}/**/*.wav', recursive=True)
        self.clean_noisy_subsample = partial(clean_noisy_subsample, sub_sample_length=sub_sample_samples)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        noisy_filename = self.filenames[idx]
        clean_filename = get_clean_name(noisy_filename)
        clean, _ = torchaudio.load(clean_filename)
        noisy, _ = torchaudio.load(noisy_filename)
        noisy, clean = self.clean_noisy_subsample(noisy[0], clean[0])
        return noisy, clean


def from_path(noisy_path, batch_size, sub_sample_length=None):
    dataset = Dataset(noisy_path, sub_sample_length)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=False)
