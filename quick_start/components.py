import logging
import warnings
from typing import Dict

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from lightning.components.python import PopenPythonScript, TracerPythonScript

logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, raise_exception=True, **kwargs)
        self.best_model_path = False

    def run(self, *args, **kwargs):
        warnings.simplefilter("ignore")
        logger.info(f"Running train_script: {self.script_path}")
        super().run(*args, **kwargs)

    def on_after_run(self, res):
        self.best_model_path = True  # Path(res["cli"].trainer.checkpoint_callback.best_model_path)


class ServeScript(PopenPythonScript):
    def __init__(self, *args, exposed_ports: Dict[str, int], **kwargs):
        assert len(exposed_ports) == 1
        super().__init__(
            *args,
            script_args=[f"--port={list(exposed_ports.values())[0]}"],
            exposed_ports=exposed_ports,
            blocking=True,
            raise_exception=True,
            **kwargs,
        )

    def run(self, checkpoint_encoding: str) -> None:
        print(" ")
        logger.info(f"Running serve_script: {self.script_path}")
        self.script_args.append("--checkpoint_path=model.pt")
        super().run()
