import argparse
import base64
import os
import torch
from io import BytesIO

import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
import torchvision.transforms as T
from PIL import Image
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train.net import Net

serve_script_path = __file__

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Server Parser")
    parser.add_argument("--checkpoint_path", type=str, help="Where to find the `checkpoint_path`")
    parser.add_argument("--port", type=int, default="8888", help="Server port`")
    hparams = parser.parse_args()

    fastapi_service = FastAPI()
    model = Net()
    checkpoint = torch.load(hparams.checkpoint_path)
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    def deserialize_image(data):
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = Image.open(buffer, mode="r")
        img = T.Compose([T.Resize((28, 28)), T.ToTensor()])(img)[0]
        return img.unsqueeze(0).unsqueeze(0)

    @fastapi_service.post("/predict")
    async def predict(request: Request):
        body = await request.json()
        data = body["payload"]["inputs"]["data"]
        img = deserialize_image(data)
        return model(img).argmax().item()

    print(f"Running the Serve Serve on port {hparams.port}")

    uvicorn.run(app=fastapi_service, host="0.0.0.0", port=hparams.port)
