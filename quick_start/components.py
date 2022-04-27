import logging
import os
import sys
import warnings
from typing import Dict
from functools import partial
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as T
import numpy as np
import base64
from time import time, sleep
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from lightning.storage import Path
from lightning.storage.path import Path
from lightning import LightningFlow
from lightning.components.python import PopenPythonScript, TracerPythonScript
from lightning.frontend import StreamlitFrontend
from lightning.utilities.state import AppState

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


class ServeScript(PopenPythonScript):
    def __init__(self, *args, exposed_ports: Dict[str, int], **kwargs):
        assert len(exposed_ports) == 1
        super().__init__(
            *args,
            script_args=[f"--port={list(exposed_ports.values())[0]}"],
            exposed_ports=exposed_ports,
            blocking=False,
            raise_exception=True,
            **kwargs,
        )

    def run(self, checkpoint_path: Path) -> None:
        logger.info(f"Running serve_script: {self.script_path}")
        self.script_args.append(f"--checkpoint_path={str(checkpoint_path)}")
        super().run()


class DemoUI(LightningFlow):

    def __init__(self):
        super().__init__()
        self.data_downloaded = False
        self.requests_count = 0
        self.serve_url = None
        self.correct = 0
        self.total = 0

    def run(self, serve_url: str):
        self.serve_url = f"{serve_url}/predict"

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


@st.experimental_memo
def load_dataset():
    dataset = MNIST("./data", train=True)
    return dataset, len(dataset)

def make_request(image, serve_url: str):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")
    body = {"session": "UUID", "payload": {"inputs": {"data": img_str}}}
    t0 = time()
    resp = requests.post(serve_url, json=body)
    t1 = time()
    return {"response": resp.json(), "request_time": t1 - t0}

def render_fn(state: AppState):
    if not state.data_downloaded:
        MNIST('./data', download=True)
        state.data_downloaded = True

    st.write(f"The *Demo Tab* is running StreamLit UI within an Iframe. Every 0.5 second, this machine makes a request with following image to the endpoint {state.serve_url} serving the previously trained model with FastAPI. The *API Tab* shows the Fast API Swagger UI associated with the Endpoint (https://swagger.io/tools/swagger-ui/).")

    correct = state.correct
    total = state.total

    if total != 0:
         st.write(f"The current model accuracy is {100 * round(correct / float(total), 2)} % with {correct} / {total} requests made.")

    dataset, L = load_dataset()
    random_idx = np.random.choice(range(L))
    image, label = dataset[random_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, width=250)

    with col2:
        if not state.serve_url:
            st.write("The Server isn't available")
            return
        try:
            response = make_request(image, state.serve_url)
        except JSONDecodeError:
            st.write("The Server isn't available")
            return

        pred = response["response"]["prediction"]
        state.total = state.total + 1
        if pred == label:
            state.correct = state.correct + 1
        state.requests_count = state.requests_count + 1
        st.write("Received Prediction")
        st.json(response)
        sleep(0.5)
        st.experimental_rerun()