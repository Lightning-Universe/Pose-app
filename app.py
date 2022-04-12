import os

from lightning import CloudCompute, LightningApp, LightningFlow

from quick_start.components import PyTorchLightningScript, ServeScript
from quick_start.serve.serve import serve_script_path
from quick_start.train.train import train_script_path


class RootFlow(LightningFlow):
    def __init__(self, use_gpu: bool = False):
        super().__init__()
        # Those are custom components for demo purposes
        # and you can modify them or create your own.
        self.train = PyTorchLightningScript(
            script_path=train_script_path,
            script_args=[
                "--trainer.max_epochs=4",
                "--trainer.limit_train_batches=4",
                "--trainer.limit_val_batches=4",
                "--trainer.callbacks=ModelCheckpoint",
                "--trainer.callbacks.monitor=val_acc",
            ],
            cloud_compute=CloudCompute("gpu" if use_gpu else "cpu", 1),
        )

        self.serve = ServeScript(
            script_path=serve_script_path,
            exposed_ports={"serving": 1111},
            cloud_compute=CloudCompute("cpu", 1),
        )

    def run(self):
        # 1. Run the ``train_script_path`` that trains a PyTorch model.
        self.train.run()

        # 2. Will be True when a checkpoint is created by the ``train_script_path``
        # and added to the train work state.
        if self.train.best_model_path:
            # 3. Serve the model until killed.
            self.serve.run(checkpoint_path=self.train.best_model_path)
            self._exit("Hello World End")

    def configure_layout(self):
        return [{"name": "API", "content": self.serve.exposed_url("serving") + "/docs"}]


app = LightningApp(RootFlow(use_gpu=bool(int(os.getenv("USE_GPU", "0")))))
