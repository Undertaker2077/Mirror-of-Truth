from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "models/retouch_detector_v2/best_model.pt"
TYPE_PATH = ROOT / "models/retouch_multitask_cnn_v1/best_model.pt"
REGION_PATH = ROOT / "models/yolo_retouch_regions_v1/best.pt"
EXPECTED_HASHES = {
    V2_PATH: "52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D",
    TYPE_PATH: "19F0D5F220A2DBE8C86E6959321FA87336475C99ABEE154E77579AC7841707BB",
    REGION_PATH: "DDD344917465425FFD15379DFC00324CFFA4126BDB41ECE4C0BBED7DF071CCDB",
}
TYPE_NAMES = ("smoothing", "whitening", "slimming")
STRENGTH_NAMES = ("none", "low", "medium", "high")


class TypeNet(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; cin=3
        for c in (24,48,96,128):
            layers += [nn.Conv2d(cin,c,3,2,1,bias=False),nn.BatchNorm2d(c),nn.SiLU(),
                       nn.Conv2d(c,c,3,1,1,groups=c,bias=False),nn.BatchNorm2d(c),nn.SiLU()];cin=c
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1));self.types=nn.Linear(128,3);self.strength=nn.Linear(128,4)
    def forward(self,x): z=self.features(x).flatten(1);return self.types(z),self.strength(z)


def _device(): return "cuda:0" if torch.cuda.is_available() else "cpu"

def _verify(path):
    if not path.is_file(): raise FileNotFoundError(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_HASHES[path]: raise RuntimeError(f"checkpoint SHA-256 mismatch: {path}")


@lru_cache(maxsize=1)
def _v2():
    from torchvision.models import efficientnet_b0
    _verify(V2_PATH);model=efficientnet_b0(weights=None);model.classifier[1]=nn.Linear(model.classifier[1].in_features,1)
    state=torch.load(V2_PATH,map_location=_device(),weights_only=True);model.load_state_dict(state["model_state"])
    return model.to(_device()).eval()


@lru_cache(maxsize=1)
def _type():
    _verify(TYPE_PATH);state=torch.load(TYPE_PATH,map_location=_device(),weights_only=True);model=TypeNet().to(_device())
    model.load_state_dict(state["model_state"]);return model.eval()


@lru_cache(maxsize=1)
def _region():
    from ultralytics import YOLO
    _verify(REGION_PATH)
    return YOLO(REGION_PATH)


def _rgb(path, size):
    bgr=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"unreadable image: {path}")
    return cv2.resize(bgr,(size,size))[:,:,::-1].copy()


def v2_predict(path):
    from PIL import Image
    from torchvision import transforms
    preprocess=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                   transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(path) as image: x=preprocess(image.convert("RGB"))[None].to(_device())
    with torch.no_grad(): p=_v2()(x).flatten()[0].sigmoid().item()
    return {"probability":p}


def type_predict(path):
    x=torch.from_numpy(_rgb(path,96)).permute(2,0,1).float()[None].div(255).to(_device())
    with torch.no_grad(): a,b=_type()(x);p=a.sigmoid()[0].cpu().tolist();s=b.argmax(1).item()
    return {"probabilities":dict(zip(TYPE_NAMES,p)),"strength":STRENGTH_NAMES[s]}


def region_predict(path, confidence=.05):
    result=_region().predict(source=str(path),imgsz=384,device=0 if torch.cuda.is_available() else "cpu",conf=confidence,verbose=False)[0]
    if result.masks is None:return []
    output=[]
    for cls,conf,poly in zip(result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.masks.xyn):
        output.append({"name":result.names[int(cls)],"confidence":round(float(conf),6),
                       "polygon":[[round(float(x),6),round(float(y),6)] for x,y in poly.tolist()]})
    return output
