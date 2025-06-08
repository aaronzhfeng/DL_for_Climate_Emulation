import os
import pandas as pd
from lightning.pytorch.callbacks import Callback
import torch

class MetricsLogger(Callback):
    """Callback to store epoch metrics into a CSV file."""

    def __init__(self, save_dir: str, filename: str = "metrics.csv"):
        super().__init__()
        self.save_dir = save_dir
        self.filename = filename
        self.records = []

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {"epoch": trainer.current_epoch}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()
            else:
                try:
                    metrics[k] = float(v)
                except Exception:
                    continue
        self.records.append(metrics)

    def on_test_end(self, trainer, pl_module):
        metrics = {"epoch": "test"}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.item()
            else:
                try:
                    metrics[k] = float(v)
                except Exception:
                    continue
        self.records.append(metrics)
        self._save()

    def on_fit_end(self, trainer, pl_module):
        self._save()

    def _save(self):
        if not self.records:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_csv(os.path.join(self.save_dir, self.filename), index=False)
