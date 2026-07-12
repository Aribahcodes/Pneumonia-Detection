from pathlib import Path
import kagglehub


def get_raw_dir():
    """Return path to raw RSNA data via kagglehub (cached after first call, so
    calling this in every notebook is cheap — no repeated downloads)."""
    path = kagglehub.competition_download('rsna-pneumonia-detection-challenge')
    return Path(path)