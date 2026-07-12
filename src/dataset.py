import numpy as np
import pydicom
from pathlib import Path
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    """DICOM-aware Dataset for the RSNA pneumonia classification task.

    Reads a cached .npy array if available (see src.preprocessing.cache_image),
    otherwise falls back to decoding the DICOM directly. Replicates the
    single grayscale channel to 3 channels for the ImageNet-pretrained
    backbone, then applies the given torchvision transform.
    """

    def __init__(self, df, image_dir, transform=None, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_path = self.cache_dir / f"{row['patientId']}.npy" if self.cache_dir else None

        if cache_path and cache_path.exists():
            img = np.load(cache_path)
        else:
            ds = pydicom.dcmread(self.image_dir / f"{row['patientId']}.dcm")
            img = ds.pixel_array

        img_3ch = np.stack([img] * 3, axis=-1)

        if self.transform:
            img_3ch = self.transform(img_3ch)

        label = row['label']
        return img_3ch, label