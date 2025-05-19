"""
ただ単に、モデルのロードに関するコードをまとめているだけ
現時点では、使い勝手やIOの共通化とかを考えているわけではない

models:
  - Segment Anything: https://github.com/facebookresearch/segment-anything
  - Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
"""

import sys
import os
import requests
from functools import lru_cache
import torch
from transformers import pipeline


BASE_PATH = os.path.expanduser("~/.cache/sj_models")


################################################################################
# Segment Anything


seg_hs = {
    "vit_h": "4b8939",
    "vit_l": "0b3195",
    "vit_b": "01ec64",
}
seg_files = {k: f"sam_{k}_{v}.pth" for k, v in seg_hs.items()}
seg_urls = {
    k: f"https://dl.fbaipublicfiles.com/segment_anything/{v}"
    for k, v in seg_files.items()
}


def _download_segment_model_parameter(model="vit_h"):
    """
    Download the SAM model parameter file.
    Parameters will be saved in `~/.cache/sj_models`.
    If the file already exists, it will not be downloaded.
    Args:
        model (str): SAM model name. Choose from "vit_h", "vit_l", "vit_b".
    """
    file, url = seg_files[model], seg_urls[model]
    path = os.path.join(BASE_PATH, file)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url)
        with open(path, "wb") as f:
            f.write(response.content)


@lru_cache(maxsize=1)
def load_segment_model(device=None, model="vit_h"):
    """
    Load the SAM model.
    """
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    print("Loading SAM model...")
    _download_segment_model_parameter(model)
    file = seg_files[model]
    path = os.path.join(BASE_PATH, file)
    sam = sam_model_registry[model](checkpoint=path)
    if device is not None:
        sam.to(device=device)
    mask_generator_ = SamAutomaticMaskGenerator(model=sam)
    return mask_generator_


################################################################################
# Depth Anything V2


@lru_cache(maxsize=1)
def load_depth_model(device=None, model="Base-hf"):
    """
    Load the depth model (pipeline).
    Args:
        model (str): SAM model name. Choose from "Small", "Base", "Large", "Small-hf", "Base-hf", "Large-hf".

    Note:
        The following notation about the pipeline is mentioned at the URL below:
        https://github.com/DepthAnything/Depth-Anything-V2,

        > Note 2: Due to the upsampling difference between OpenCV (we used) and Pillow (HF used),
        > predictions may differ slightly.
        > So you are more recommended to use our models through the way introduced above (without pipeline).
        However, for convenience, we are using the method with the pipeline.
    """

    print("Loading DepthAnythingV2 model...")
    model_path = f"depth-anything/Depth-Anything-V2-{model}"
    if device is None:
        device = "cpu"
    pipe = pipeline(task="depth-estimation", model=model_path, device=device)
    return pipe
