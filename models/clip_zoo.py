"""Minimal model loader registry for CLIP-family backbones."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    key: str
    arch: str
    source: str


MODEL_ZOO: Dict[str, ModelSpec] = {
    "clip_openai_vitl14": ModelSpec("clip_openai_vitl14", "ViT-L/14", "openai"),
    "openclip_laion_vitl14": ModelSpec("openclip_laion_vitl14", "ViT-L/14", "laion2b"),
    "siglip2_vitl16": ModelSpec("siglip2_vitl16", "ViT-L/16", "webli"),
    "eva02_clip_l14": ModelSpec("eva02_clip_l14", "EVA02-L/14", "merged-2b"),
    "dfn_clip_vitl14": ModelSpec("dfn_clip_vitl14", "ViT-L/14", "dfn-2b"),
}


def list_models() -> Dict[str, ModelSpec]:
    return MODEL_ZOO
