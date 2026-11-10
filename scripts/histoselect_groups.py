"""HistoSelect inputs from cached CONCH features (no re-extraction): per-slide tissue groups and the question.

    python scripts/histoselect_groups.py --config configs/server.yaml

Needs the CONCH environment (``conch`` package) and the CONCH weights in HF_HOME. As in the official
``data_preprocessing/tissue_segmentation.py`` (github.com/winston52/HistoSelect @ e4bb3c1): 13 tissue
classes x 13 prompt templates, text embeddings averaged per class and normalised; each patch's group = argmax
similarity to the class embeddings. Image-text patch embeddings = cached pre-projection CONCH features @
``visual.proj_contrast`` (the cache stores ``encode_image(proj_contrast=False)``). The question embedding is
``encode_text`` of the fixed question (``generate_question_embeddings.py``). Writes
CACHE_ROOT/histoselect/<slide>.npy (int8 group per cached row, same row order), question.npy,
group_counts.json and run.json. Finished slides are skipped.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from lesionbench.config import load_config, write_run_json

CLASSES = ["malignant tissue", "benign tissue", "stroma", "lymphocytes", "necrosis", "adipose tissue",
           "tissue artifact", "blood vessel", "extracellular mucin", "nerve", "hemorrhage", "smooth muscle",
           "plasma cells"]
TEMPLATES = ["an H&E stained image of CLASSNAME.", "a photomicrograph showing CLASSNAME.",
             "tissue section showing CLASSNAME.", "area dominated by CLASSNAME.",
             "patch with abundant CLASSNAME.",
             "region with evidence of CLASSNAME.", "an example of CLASSNAME.", "this is CLASSNAME.",
             "presence of CLASSNAME.", "CLASSNAME is present.", "an image of CLASSNAME.",
             "a histopathological photograph of CLASSNAME.", "a histopathological image of CLASSNAME."]
QUESTION = "Is there metastatic carcinoma in this lymph node?"


def conch_tokenize(tokenizer, texts: list[str]) -> torch.Tensor:
    """``conch.open_clip_custom.tokenize`` with ``tokenizer(...)`` instead of ``batch_encode_plus`` (removed
    in transformers 5): same arguments (max_length 127, padding to max length, truncation), then one <pad>."""
    ids = tokenizer(texts, max_length=127, add_special_tokens=True, return_token_type_ids=False,
                    truncation=True, padding="max_length", return_tensors="pt")["input_ids"]
    return torch.nn.functional.pad(ids, (0, 1), value=tokenizer.pad_token_id)


def tissue_groups(feats: np.ndarray, proj: torch.Tensor, class_emb: torch.Tensor) -> np.ndarray:
    """Group index per patch: argmax cosine similarity of the projected patch embedding to each class."""
    z = torch.nn.functional.normalize(torch.from_numpy(feats).float() @ proj, dim=-1)
    return (z @ class_emb.T).argmax(dim=1).numpy().astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache = Path(cfg["paths"]["cache_root"])
    out = cache / "histoselect"
    out.mkdir(parents=True, exist_ok=True)

    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download("MahmoodLab/CONCH", "pytorch_model.bin")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=weights)
    model.eval()
    tok = get_tokenizer()
    with torch.no_grad():
        prompts = [t.replace("CLASSNAME", c) for c in CLASSES for t in TEMPLATES]
        text = model.encode_text(conch_tokenize(tok, prompts))
        text = torch.nn.functional.normalize(text, dim=-1)
        text = text.reshape(len(CLASSES), len(TEMPLATES), -1).mean(dim=1)
        class_emb = torch.nn.functional.normalize(text, dim=-1)
        question = model.encode_text(conch_tokenize(tok, [QUESTION]))[0]
        proj = model.visual.proj_contrast.detach().float()
    np.save(out / "question.npy", question.numpy().astype(np.float32))

    counts = Counter()
    slides = sorted((cache / "conch").glob("*.h5"))
    for p in slides:
        dst = out / f"{p.stem}.npy"
        if dst.exists():
            groups = np.load(dst)
        else:
            with h5py.File(p) as f:
                groups = tissue_groups(f["features"][:], proj, class_emb)
            np.save(dst, groups)
        counts.update(groups.tolist())
    summary = {CLASSES[k]: int(v) for k, v in sorted(counts.items())}
    (out / "group_counts.json").write_text(json.dumps(summary, indent=1))
    write_run_json(out, cfg, "run.json", question=QUESTION, n_slides=len(slides), classes=CLASSES)
    print(f"{len(slides)} slides -> {out}; patches per group: {summary}")


if __name__ == "__main__":
    main()
