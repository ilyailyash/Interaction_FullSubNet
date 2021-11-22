import os

from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio


class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, sr=16000, sub_sample_length=None, hop_length=255):
        self.clean_path = self.get_clean(data_dir)
        self.noisy_path = self.get_noisy(data_dir)
        self.sr = sr
        self.sub_sample_length = sub_sample_length
        self.hop_length = hop_length



    def get_clean(self, root):
        clean_dir = os.path.join(root, 'clean_testset_wav')
        filenames = glob(f'{clean_dir}/*.wav', recursive=True)
        return filenames


    def get_noisy(self, root):
        noisy_dir = os.path.join(root, 'noisy_testset_wav')
        filenames = glob(f'{noisy_dir}/*.wav', recursive=True)
        return filenames

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        clean = torchaudio.load(self.clean_path[idx])[0]
        noisy = torchaudio.load(self.noisy_path[idx])[0]

        # noisy = self.normalize(noisy)
        length = clean.size(-1)
        clean.squeeze_(0)
        noisy.squeeze_(0)

        if self.sub_sample_length:
            сut_len = self.sub_sample_length*self.sr
            start = torch.randint(0, length - сut_len - 1, (1, ))
            end = start + сut_len
            clean = clean[start:end]
            noisy = noisy[start:end]

        return noisy, clean

class DemandDatasetInference(Dataset):
    def __init__(self, noisy_dataset, sr=16000, sub_sample_length=2, hop_length=255, limit=True, offset=0):
        # self.clean_path = self.get_clean(data_dir)
        self.noisy_path = self.get_noisy(noisy_dataset)
        self.sr = sr
        self.sub_sample_length = sub_sample_length
        self.hop_length = hop_length

        self.resample = torchaudio.transforms.Resample(48000, 16000)

    def get_clean(self, root):
        clean_dir = os.path.join(root, 'clean_testset_wav')
        filenames = glob(f'{clean_dir}/*.wav', recursive=True)
        return filenames

    def get_noisy(self, root):
        noisy_dir = os.path.join(root, 'noisy_testset_wav')
        filenames = glob(f'{noisy_dir}/*.wav', recursive=True)
        return filenames

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        noisy = torchaudio.load(self.noisy_path[idx])[0]
        noisy = self.resample(noisy)

        # noisy = self.normalize(noisy)
        noisy.squeeze_(0)

        return noisy, os.path.splitext(os.path.basename(self.noisy_path[idx]))[0]