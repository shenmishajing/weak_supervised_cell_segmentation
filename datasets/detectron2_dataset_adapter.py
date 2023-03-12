import os
from collections import OrderedDict
from typing import Sequence

os.environ["DETECTRON2_DATASETS"] = "data"

import copy

from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.build import (
    _test_loader_from_config,
    _train_loader_from_config,
    trivial_batch_collator,
)
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
)
from torch.utils.data import BatchSampler, DataLoader

from datasets.base import LightningDataModule


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self, dataset, sampler, batch_size, drop_last) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = dataset
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets = [[] for _ in range(2)]

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.dataset[idx]
            width, height = data_info["width"], data_info["height"]
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[: self.batch_size]
                left_data = left_data[self.batch_size :]


class Detectron2DataSetAdapter(LightningDataModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self.split_names:
            cfg = get_cfg()
            cfg.merge_from_other_cfg(CfgNode(self.dataset_cfg[name]))
            self.dataset_cfg[name] = cfg

        self.evaluators = {}

    def _build_dataset(self, split):
        def _get_dataset(dataset):
            mapper = dataset.pop("mapper")
            dataset["dataset"] = MapDataset(
                DatasetFromList(dataset["dataset"], copy=False), mapper
            )
            return dataset

        if split == "train":
            dataset = _get_dataset(_train_loader_from_config(self.dataset_cfg[split]))
        else:
            dataset = []
            for dataset_name in self.dataset_cfg[split].DATASETS.TEST:
                dataset.append(
                    _get_dataset(
                        _test_loader_from_config(self.dataset_cfg[split], dataset_name)
                    )
                )
        self.datasets[split] = dataset

    def _build_collate_fn(self, collate_fn_cfg):
        return trivial_batch_collator

    def _handle_batch_sampler(self, dataloader_cfg, dataset, split="train"):
        if "aspect_ratio_grouping" in dataset and dataset["aspect_ratio_grouping"]:
            dataloader_cfg["batch_sampler"] = AspectRatioBatchSampler(
                dataset["dataset"],
                self._build_sampler(dataloader_cfg, dataset["dataset"]),
                dataloader_cfg.pop("batch_size", 1),
                dataloader_cfg.pop("drop_last", False),
            )
        return dataloader_cfg

    def _construct_dataloader(self, dataset, split="train", set_batch_size=False):
        dataloader_cfg = copy.deepcopy(self.dataloader_cfg.get(split, {}))
        if set_batch_size:
            dataloader_cfg["batch_size"] = self.batch_size
        dataloader_cfg["collate_fn"] = self._build_collate_fn(
            dataloader_cfg.get("collate_fn", {})
        )
        return DataLoader(
            dataset["dataset"],
            **self._handle_batch_sampler(dataloader_cfg, dataset, split=split)
        )

    def _build_dataloader(self, dataset, *args, **kwargs):
        if isinstance(dataset, Sequence):
            return [self._build_dataloader(ds, *args, **kwargs) for ds in dataset]
        else:
            return self._construct_dataloader(dataset, *args, **kwargs)

    def process_dataset_evaluation_results(self, split="val") -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.evaluators[split]):
            results[dataset_name] = self.evaluators[split][idx].evaluate()

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def reset_dataset_evaluators(self):
        for split in self.split_names:
            self.evaluators[split] = []
            for dataset_name in self.dataset_cfg[split].DATASETS.TEST:
                evaluator = build_evaluator(
                    self.dataset_cfg[split],
                    dataset_name,
                    os.path.join(self.trainer.log_dir, "inference"),
                )
                evaluator.reset()
                self.evaluators[split].append(evaluator)
