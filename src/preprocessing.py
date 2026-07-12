import numpy as np
import pandas as pd
import pydicom
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Label mapping (Phase 1)
# ---------------------------------------------------------------------------

# Canonical 3-class mapping for this project.
# Reused across preprocessing, dataset building, training, and evaluation
# so the label integers stay consistent everywhere.
CLASS_MAP = {
    'Normal': 0,
    'No Lung Opacity / Not Normal': 1,
    'Lung Opacity': 2
}


def resolve_patient_labels(merged_df):
    """Collapse multi-box rows into one row per patientId.

    RSNA's stage_2_train_labels.csv has one row per bounding box, so
    patients with multiple pneumonia opacities appear multiple times.
    label/class_name are identical across a patient's rows, so we just
    take the first; n_boxes counts how many real (non-NaN) boxes each
    patient had, mainly useful for EDA/Grad-CAM later.
    """
    return merged_df.groupby('patientId').agg(
        label=('label', 'first'),
        class_name=('class', 'first'),
        n_boxes=('x', lambda x: x.notna().sum())
    ).reset_index()


# ---------------------------------------------------------------------------
# Image caching (Phase 2)
# ---------------------------------------------------------------------------

def cache_image(patient_id, image_dir, cache_dir, size=224):
    """Decode a DICOM once, resize, and save as .npy. Skips if already cached.

    Caching the raw decoded/resized array (not the augmented tensor) avoids
    repeated pydicom decode cost on every epoch, while still letting
    augmentation stay random per-epoch at train time.
    """
    cache_path = Path(cache_dir) / f"{patient_id}.npy"
    if cache_path.exists():
        return cache_path

    ds = pydicom.dcmread(Path(image_dir) / f"{patient_id}.dcm")
    img = ds.pixel_array
    img_resized = np.array(Image.fromarray(img).resize((size, size)))
    np.save(cache_path, img_resized)
    return cache_path


# ---------------------------------------------------------------------------
# Transforms (Phase 2)
# ---------------------------------------------------------------------------

# Standard ImageNet normalization stats, since the model uses an
# ImageNet-pretrained DenseNet121 backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# No augmentation — used for validation/test so evaluation is deterministic.
val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Light, anatomically-safe augmentation — train split only.
# Deliberately excludes vertical flip (anatomically implausible) and
# heavy rotation/crop (risks cutting off or distorting the pathology).
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])