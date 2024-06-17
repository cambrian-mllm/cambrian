"""Callback to alert when training loss goes to NaN or Inf"""

from typing import List

import numpy as np
from transformers.integrations import WandbCallback
from ezcolorlog import root_logger as logger


class NanInfAlertWandbCallback(WandbCallback):
    def __init__(self, metrics: List[str] = ["loss"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = metrics

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs, **kwargs)

        if (
            logs is None or
            self._wandb is None or
            not self._initialized
        ):
            return

        # now check for NaN or Inf
        for metric in self.metrics:
            metric_val = logs.get(metric, None)
            if metric_val is None:
                continue

            if np.isnan(metric_val) or np.isinf(metric_val):
                logger.error(f"{metric} is {metric_val}")
                if state.is_world_process_zero:
                    self._wandb.alert(title="NaN or Inf detected", text=f"{metric} is {metric_val}", level="WARN")
                raise RuntimeError(f"{metric} is {metric_val}")
