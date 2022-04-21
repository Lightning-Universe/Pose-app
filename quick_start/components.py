import logging
import os
import sys
import warnings
from typing import Dict
from functools import partial

from lightning.storage import Path

sys.path.insert(0, os.path.dirname(__file__))

from lightning.storage.path import Path
from lightning.components.python import PopenPythonScript, TracerPythonScript

logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, raise_exception=True, **kwargs)
        self.best_model_path = None
        self.run_url = None

    def configure_tracer(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback
        from pytorch_lightning.loggers import WandbLogger

        tracer = super().configure_tracer()

        class CollectWandbURL(Callback):

            def __init__(self, work):
                self._work = work

            def on_train_start(self, trainer, *_):
                self._work.run_url = trainer.logger.experiment._settings.run_url

        def trainer_pre_fn(self, *args, work=None, **kwargs):
            kwargs['callbacks'].append(CollectWandbURL(work))
            kwargs['logger'] = [WandbLogger(save_dir=os.path.dirname(__file__))]
            return {}, args, kwargs

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=partial(trainer_pre_fn, work=self))
        return tracer

    def run(self, *args, **kwargs):
        warnings.simplefilter("ignore")
        logger.info(f"Running train_script: {self.script_path}")
        super().run(*args, **kwargs)

    def on_after_run(self, res):
        self.best_model_path = Path(res["cli"].trainer.checkpoint_callback.best_model_path)


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

    def run(self, checkpoint_path: Path) -> None:
        logger.info(f"Running serve_script: {self.script_path}")
        self.script_args.append(f"--checkpoint_path={str(checkpoint_path)}")
        super().run()
