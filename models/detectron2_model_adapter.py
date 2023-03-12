from typing import List

from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import EventStorage

from .base import LightningModule


class Detectron2ModelAdapter(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = get_cfg()
        self.cfg.merge_from_other_cfg(CfgNode(cfg))
        self.model = META_ARCH_REGISTRY.get(self.cfg.MODEL.META_ARCHITECTURE)(self.cfg)
        self.storage = None

    def forward(self, batch):
        return self.model(batch)

    def forward_step(
        self, batch, *args, dataloader_idx: int = 0, split="val", **kwargs
    ):
        if not isinstance(batch, List):
            batch = [batch]
        outputs = self(batch)
        self.trainer.datamodule.evaluators[split][dataloader_idx].process(
            batch, outputs
        )
        return outputs

    def forward_epoch_end(self, outputs, *args, split="val", **kwargs):
        results = self.trainer.datamodule.process_dataset_evaluation_results(split)
        self.log_dict(self.flatten_dict(results, split), sync_dist=True)
        return results

    def on_train_epoch_start(self, *args, **kwargs):
        if self.storage is None:
            self.storage = EventStorage(0)
        self.storage.__enter__()

    def training_epoch_end(self, *args, **kwargs):
        self.storage.__exit__(None, None, None)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.trainer.datamodule.reset_dataset_evaluators()

    def on_test_epoch_start(self, *args, **kwargs):
        self.trainer.datamodule.reset_dataset_evaluators()
