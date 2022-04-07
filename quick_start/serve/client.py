import base64
from pathlib import Path

import requests

with Path("REPLACE_WITH_YOUR_IMAGE").open("rb") as f:
    imgstr = base64.b64encode(f.read()).decode("UTF-8")

body = {"session": "UUID", "payload": {"inputs": {"data": imgstr}}}
resp = requests.post("http://127.0.0.1:8888/predict", json=body)
print(resp.json())
