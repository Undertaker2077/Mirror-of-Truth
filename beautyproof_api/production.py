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
TYPE_PATH = ROOT / "models/retouch_three_type_v1/best_model.pt"
REGION_PATH = ROOT / "models/yolo_retouch_regions_v1/best.pt"
EXPECTED_HASHES = {
    V2_PATH: "52F38353CEB4F20325B8AF84C0A0973FD48FEB57323B4429465D5C10FCFDC94D",
    TYPE_PATH: "179A317C8D77961131CC7544F97B0CD9789CD9C02862A5CA95986893983CEC52",
    REGION_PATH: "DDD344917465425FFD15379DFC00324CFFA4126BDB41ECE4C0BBED7DF071CCDB",
}
TYPE_NAMES = ("skin_enhancement", "face_slimming", "eye_enlargement", "facial_contouring")
PUBLISHED_TYPE_NAMES = ("skin_enhancement", "face_slimming", "facial_contouring")


class TypeNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import efficientnet_b0
        base=efficientnet_b0(weights=None);self.features=base.features;self.pool=base.avgpool
        width=base.classifier[1].in_features;self.dropout=nn.Dropout(.25)
        self.retouch_head=nn.Linear(width,1);self.type_head=nn.Linear(width,4);self.strength_head=nn.Linear(width,4)
    def forward(self,x):
        z=self.dropout(torch.flatten(self.pool(self.features(x)),1))
        return self.retouch_head(z),self.type_head(z),torch.sigmoid(self.strength_head(z))


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
    model.load_state_dict(state["model"]);model.thresholds=state["thresholds"];return model.eval()


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
    from PIL import Image
    from torchvision import transforms
    prep=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(path) as image:x=prep(image.convert("RGB"))[None].to(_device())
    model=_type()
    with torch.no_grad(): _,logits,strength=model(x);p=logits.sigmoid()[0].cpu().tolist();s=strength[0].cpu().tolist()
    return {"probabilities":{n:p[TYPE_NAMES.index(n)] for n in PUBLISHED_TYPE_NAMES},
            "strengths":{n:s[TYPE_NAMES.index(n)] for n in PUBLISHED_TYPE_NAMES},
            "thresholds":{n:float(model.thresholds[n]) for n in PUBLISHED_TYPE_NAMES}}


def region_predict(path, confidence=.05):
    result=_region().predict(source=str(path),imgsz=384,device=0 if torch.cuda.is_available() else "cpu",conf=confidence,verbose=False)[0]
    if result.masks is None:return []
    output=[]
    for cls,conf,poly in zip(result.boxes.cls.tolist(),result.boxes.conf.tolist(),result.masks.xyn):
        output.append({"name":result.names[int(cls)],"confidence":round(float(conf),6),
                       "polygon":[[round(float(x),6),round(float(y),6)] for x,y in poly.tolist()]})
    return output
