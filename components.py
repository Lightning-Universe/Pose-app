import codecs
import os
import pickle
from typing import Dict

from lightning.components.python import PopenPythonScript, TracerPythonScript


def get_train_script_path() -> str:
    return os.path.join(os.path.dirname(__file__), "train/train.py")


def get_serve_script_path() -> str:
    return os.path.join(os.path.dirname(__file__), "serve/serve.py")


class PyTorchLightningCLIScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_path = None

    def on_after_run(self, res):
        import torch

        obj = torch.load(res["cli"].trainer.checkpoint_callback.best_model_path)
        self.best_model_path = codecs.encode(pickle.dumps(obj), "base64").decode()


class ServePythonScript(PopenPythonScript):
    def __init__(self, *args, exposed_ports: Dict[str, int], **kwargs):
        assert len(exposed_ports) == 1
        super().__init__(
            *args,
            script_args=[f"--port={list(exposed_ports.values())[0]}"],
            exposed_ports=exposed_ports,
            blocking=True,
            **kwargs,
        )

    def run(self, checkpoint_encoding: str) -> None:
        import torch

        unpickled = pickle.loads(codecs.decode(checkpoint_encoding.encode(), "base64"))
        torch.save(unpickled, "model.pt")
        self.script_args.append("--checkpoint_path=model.pt")
        super().run()
