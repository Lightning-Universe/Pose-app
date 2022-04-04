import argparse
import base64
import os
from io import BytesIO

import torch
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from PIL import Image

from examples.hello_world.train.net import Net


def main():

    parser = argparse.ArgumentParser("Server Parser")
    parser.add_argument("--checkpoint_path", type=str, help="Where to find the `checkpoint_path`")
    parser.add_argument("--port", type=int, default="8888", help="Server port`")
    hparams = parser.parse_args()

    fastapi_service = FastAPI()
    model = Net()

    if not os.path.exists(str(hparams.checkpoint_path)):
        raise Exception(f"The checkpoint path {hparams.checkpoint_path} doesn't exists.")

    state_dict = torch.load(hparams.checkpoint_path)["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

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


if __name__ == "__main__":
    main()
