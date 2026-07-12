import pandas as pd

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