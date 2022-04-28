import logging
import os
import warnings
from functools import partial
import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from lightning.storage import Path
from lightning.components.python import TracerPythonScript
from lightning.components.serve import ServeGradio
import gradio as gr

logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, raise_exception=True, **kwargs)
        self.best_model_path = None
        self.run_url = ""

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
        lightning_module = res["cli"].trainer.lightning_module
        checkpoint = torch.load(res["cli"].trainer.checkpoint_callback.best_model_path)
        lightning_module.load_state_dict(checkpoint["state_dict"])
        lightning_module.to_torchscript("model_weight.pt")
        self.best_model_path = Path("model_weight.pt")

class ImageServeGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", shape=(28, 28))
    outputs = gr.outputs.Label(num_top_classes=10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.examples = [os.path.join("./images", f) for f in os.listdir("./images")]
        self.best_model_path = None
        self._transform = None
        self._labels = {idx: str(idx) for idx in range(10)}

    def run(self, best_model_path):
        self.best_model_path = best_model_path
        self._transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
        super().run()

    def predict(self, img):
        img = self._transform(img)[0]
        img = img.unsqueeze(0).unsqueeze(0)
        prediction = torch.exp(self.model(img))
        return {self._labels[i]: prediction[0][i].item() for i in range(10)}

    def build_model(self):
        model = torch.load(self.best_model_path)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model