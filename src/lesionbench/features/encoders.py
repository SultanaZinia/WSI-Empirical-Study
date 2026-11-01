"""Frozen patch encoders: name -> (embed fn on a (B, 3, H, W) batch, image transform, feature dim).

UNI (v1) follows mahmoodlab/UNI get_encoder('uni'); CONCH follows the mahmoodlab/CONCH README
(encode_image with proj_contrast=False, normalize=False: the pre-projection embedding meant for MIL).
Weights come from the Hugging Face cache (HF_HOME); jobs run with HF_HUB_OFFLINE=1, so download them first
with scripts/download_weights.py on a login node. ``dummy`` needs no weights and is for tests.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision import transforms

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HF_REPOS = {
    "uni": ("MahmoodLab/UNI", "pytorch_model.bin"),
    "conch": ("MahmoodLab/CONCH", "pytorch_model.bin"),
}


def load_encoder(name: str, device="cpu") -> tuple[Callable[[Tensor], Tensor], Callable, int]:
    if name == "uni":
        import timm
        from huggingface_hub import hf_hub_download

        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
                                  num_classes=0, dynamic_img_size=True)
        state = torch.load(hf_hub_download(*HF_REPOS["uni"]), map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(*IMAGENET)])
        return model, tf, 1024
    if name == "conch":
        from conch.open_clip_custom import create_model_from_pretrained
        from huggingface_hub import hf_hub_download

        model, preprocess = create_model_from_pretrained("conch_ViT-B-16",
                                                         checkpoint_path=hf_hub_download(*HF_REPOS["conch"]))
        model.eval().to(device)
        return (lambda x: model.encode_image(x, proj_contrast=False, normalize=False)), preprocess, 512
    if name == "dummy":
        return _dummy_embed, transforms.ToTensor(), 6
    raise ValueError(f"unknown encoder {name!r}")


def _dummy_embed(x: Tensor) -> Tensor:
    """Per-channel mean and std of the image: deterministic, weight-free, 6-d."""
    return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)
