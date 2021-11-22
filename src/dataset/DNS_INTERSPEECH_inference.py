import os
from pathlib import Path

import librosa
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, noisy_dataset, limit, offset, sr):
        """
        Args:
            noisy_dataset (str): noisy dir (wav format files) or noisy filenames list
        """
        noisy_wav_files = []

        for dataset_dir in noisy_dataset:
            dataset_dir = Path(dataset_dir).expanduser().absolute()
            noisy_wav_files += librosa.util.find_files(dataset_dir.as_posix())

        print(f"Num of noisy files in {noisy_dataset}: {len(noisy_wav_files)}")

        self.length = len(noisy_wav_files)
        self.noisy_wav_files = noisy_wav_files
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        noisy_path = self.noisy_wav_files[item]
        basename = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy = librosa.load(noisy_path, sr=self.sr)[0]

        return noisy, basename
