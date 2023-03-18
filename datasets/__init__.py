import os

os.environ["DETECTRON2_DATASETS"] = "data"
from . import datasets as datasets
from .base import LightningDataModule as LightningDataModule
