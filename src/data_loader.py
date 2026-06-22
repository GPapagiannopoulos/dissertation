import os
from pathlib import Path

from dotenv import load_dotenv
from pyhealth.datasets import MIMIC4Dataset

load_dotenv()
DATA_PATH = Path(os.getenv("DATA_DIRECTORY"))
mimic4_dataset = MIMIC4Dataset(DATA_PATH)
print(mimic4_dataset.dataset_name)
