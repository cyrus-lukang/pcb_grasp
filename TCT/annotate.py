"""Minimal Grounding DINO wrapper used by annotate_tray_crop_gdino.py.

Default backend: local X-AnyLabeling ONNX (no HF download).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

DEFAULT_ONNX = (
    "/home/ubuntu/xanylabeling_data/models/"
    "groundingdino_swint_sam2_large-r20240806/groundingdino_swint_ogc_quant.onnx"
)
DEFAULT_TOKENIZER = (
    "/home/ubuntu/workspace/czh/data_flow/X-AnyLabeling/"
    "anylabeling/services/auto_labeling/configs/bert/bert_base_uncased_tokenizer.json"
)

# name -> (backend, model_path_or_id)
MODELS = {
    "gdino-base": ("grounding_dino", DEFAULT_ONNX),
    "gdino-onnx": ("grounding_dino", DEFAULT_ONNX),
}

SPECIAL_TOKENS = [101, 102, 1012, 1029]
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ClassConfig:
    name: str
    prompts: List[str]
    threshold: float = 0.25


@dataclass
class Detection:
    bbox: List[float]  # xyxy in image / crop pixels
    score: float
    label: str = ""


def _get_caption(text_prompt: str) -> str:
    caption = text_prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."
    return caption


def _generate_masks(tokenized: dict) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for tok in SPECIAL_TOKENS:
        special_tokens_mask |= input_ids == tok
    idxs = np.argwhere(special_tokens_mask)
    attention_mask = np.eye(num_token, dtype=bool).reshape(1, num_token, num_token)
    attention_mask = np.tile(attention_mask, (bs, 1, 1))
    position_ids = np.zeros((bs, num_token), dtype=int)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[
                row, previous_col + 1 : col + 1, previous_col + 1 : col + 1
            ] = True
            position_ids[row, previous_col + 1 : col + 1] = np.arange(
                0, col - previous_col
            )
        previous_col = col
    return attention_mask, position_ids


def _encode_text(text_prompt: str, tokenizer, max_text_len: int = 256):
    caption = _get_caption(text_prompt)
    raw = tokenizer.encode(caption)
    tokenized = {
        "input_ids": np.array([raw.ids], dtype=np.int64),
        "token_type_ids": np.array([raw.type_ids], dtype=np.int64),
        "attention_mask": np.array([raw.attention_mask]),
    }
    text_masks, position_ids = _generate_masks(tokenized)
    if text_masks.shape[1] > max_text_len:
        text_masks = text_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        for k in tokenized:
            tokenized[k] = tokenized[k][:, :max_text_len]
    return tokenized, text_masks, position_ids, caption


def _preprocess_image(image_rgb: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    image = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGE_MEAN) / IMAGE_STD
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, 0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rescale_boxes(boxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    converted = []
    for box in boxes:
        b = box * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        b[:2] -= b[2:] / 2
        b[2:] += b[:2]
        converted.append(b)
    if not converted:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(converted, dtype=np.float32)


class GroundingDINO:
    """ONNX Grounding DINO (swint_ogc quant)."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ONNX,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER,
        input_width: int = 1200,
        input_height: int = 800,
        providers: Sequence[str] | None = None,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"GDINO ONNX not found: {model_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"BERT tokenizer not found: {tokenizer_path}")

        if providers is None:
            avail = ort.get_available_providers()
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in avail
            ]
        self.session = ort.InferenceSession(str(model_path), providers=list(providers))
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.input_size = (int(input_width), int(input_height))
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        img = _preprocess_image(rgb, self.input_size)
        tokenized, text_masks, position_ids, caption = _encode_text(
            text_prompt, self.tokenizer
        )
        feeds = {
            "img": img,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"].astype(bool),
            "position_ids": position_ids.astype(np.int64),
            "token_type_ids": tokenized["token_type_ids"],
            "text_token_mask": text_masks,
        }
        # Only pass names the session expects
        feeds = {k: v for k, v in feeds.items() if k in self.input_names}
        outs = self.session.run(self.output_names, feeds)
        out_map = dict(zip(self.output_names, outs))
        logits = out_map.get("logits", outs[0])
        boxes = out_map.get("boxes", outs[1])

        logits_filt = _sigmoid(np.squeeze(logits, 0))
        boxes_filt = np.squeeze(boxes, 0)
        keep = logits_filt.max(axis=1) > box_threshold
        logits_filt = logits_filt[keep]
        boxes_filt = boxes_filt[keep]
        if len(boxes_filt) == 0:
            return []

        boxes_xyxy = _rescale_boxes(boxes_filt, h, w)
        dets: List[Detection] = []
        for box, logit in zip(boxes_xyxy, logits_filt):
            score = float(logit.max())
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append(Detection(bbox=[x1, y1, x2, y2], score=score, label=caption))
        return dets

    def detect_class(
        self,
        image: Image.Image,
        cc: ClassConfig,
        text_threshold: float = 0.25,
    ) -> List[Detection]:
        """Run each prompt; keep boxes above class threshold."""
        all_dets: List[Detection] = []
        for prompt in cc.prompts:
            dets = self.detect(
                image,
                prompt,
                box_threshold=float(cc.threshold),
                text_threshold=float(text_threshold),
            )
            for d in dets:
                d.label = cc.name
                all_dets.append(d)
        return all_dets
