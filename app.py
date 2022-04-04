from examples.hello_world.components import (
    get_serve_script_path,
    get_train_script_path,
    PyTorchLightningCLIScript,
    ServePythonScript,
)
from lightning import CloudCompute, LightningApp, LightningFlow


class HelloWorld(LightningFlow):
    def __init__(self):
        super().__init__()
        self.train = PyTorchLightningCLIScript(
            script_path=get_train_script_path(),
            script_args=[
                "--trainer.max_epochs=5",
                "--trainer.limit_train_batches=4",
                "--trainer.limit_val_batches=4",
                "--trainer.callbacks=ModelCheckpoint",
                "--trainer.callbacks.monitor=val_acc",
            ],
            cloud_compute=CloudCompute("cpu", 1),
        )

        self.serve = ServePythonScript(
            script_path=get_serve_script_path(),
            exposed_ports={"serving": 8888},
            cloud_compute=CloudCompute("cpu", 1),
        )

    def run(self):
        self.train.run()
        if self.train.best_model_path is not None:
            self.serve.run(self.train.best_model_path)  # runs until killed.
            self._exit("Hello World End")

    def configure_layout(self):
        return [{"name": "API Access", "content": self.serve.exposed_url("serving") + "/docs"}]


app = LightningApp(HelloWorld())
