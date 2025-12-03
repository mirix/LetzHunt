import argparse
from dataclasses import dataclass
from datetime import datetime, date as _date
import re

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration


# ---------------------------
# OCR utilities
# ---------------------------

def clean_repeated_substrings(text: str) -> str:
    """Clean repeated substrings in text (your original logic)."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]

    return text


@dataclass
class Line:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


# TEXT(x1,y1),(x2,y2) where x,y are in [0,1000] normalized coordinates
PATTERN = re.compile(r"""
    (?P<text>.+?)
    \(
      (?P<x1>\d+)
      ,
      (?P<y1>\d+)
    \),
    \(
      (?P<x2>\d+)
      ,
      (?P<y2>\d+)
    \)
""", re.VERBOSE)


def parse_compact_ocr_string(s: str, img_w: int, img_h: int) -> list[Line]:
    """
    Parse HunyuanOCR's compact output string and
    denormalize coordinates from [0,1000] to image pixels.
    """
    lines: list[Line] = []
    for m in PATTERN.finditer(s):
        text = m.group("text").strip()
        x1_n = float(m.group("x1"))
        y1_n = float(m.group("y1"))
        x2_n = float(m.group("x2"))
        y2_n = float(m.group("y2"))

        # denormalize (same logic as denormalize_coordinates in utils.py)
        x1 = int(x1_n * img_w / 1000.0)
        y1 = int(y1_n * img_h / 1000.0)
        x2 = int(x2_n * img_w / 1000.0)
        y2 = int(y2_n * img_h / 1000.0)

        lines.append(Line(text, x1, y1, x2, y2))
    return lines


# ---------------------------
# Geometry-based clustering in *pixel* space
# ---------------------------

def lines_are_close(a: Line, b: Line,
                    max_dx: float, max_dy: float) -> bool:
    dx = abs(a.cx - b.cx)
    dy = abs(a.cy - b.cy)
    return dx <= max_dx and dy <= max_dy


def cluster_lines_into_labels(lines: list[Line], img_w: int, img_h: int) -> list[list[Line]]:
    """
    Cluster lines into labels based on spatial proximity in pixel space.
    Each cluster should correspond to one hunting-lot label.
    """
    if not lines:
        return []

    # thresholds as fractions of image size; tune as needed
    max_dx = img_w * 0.2   # horizontally near
    max_dy = img_h * 0.2   # vertically near

    print(f"[DEBUG] img_w={img_w}, img_h={img_h}, max_dx={max_dx}, max_dy={max_dy}")

    labels: list[list[Line]] = []
    visited: set[int] = set()

    for i, line in enumerate(lines):
        if i in visited:
            continue

        cluster_idx = len(labels)
        labels.append([])
        stack = [i]
        visited.add(i)

        while stack:
            idx = stack.pop()
            l = lines[idx]
            labels[cluster_idx].append(l)

            for j, other in enumerate(lines):
                if j in visited:
                    continue
                if lines_are_close(l, other, max_dx, max_dy):
                    visited.add(j)
                    stack.append(j)

    for label in labels:
        label.sort(key=lambda l: (l.cy, l.cx))

    return labels


# ---------------------------
# Block building (lot + Battue/Treibjagd + dates)
# ---------------------------

LOT_NUMBER_RE = re.compile(r"^\d{1,4}$")
DATE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$")


def is_lot_number(text: str) -> bool:
    return bool(LOT_NUMBER_RE.fullmatch(text.strip()))


def is_battue_label(text: str) -> bool:
    t = text.lower()
    return "battue" in t and "treibjagd" in t


def is_date_line(text: str) -> bool:
    return bool(DATE_RE.fullmatch(text))


def build_blocks_from_labels(labels: list[list[Line]]):
    """
    From clusters of lines, build structured blocks:
    lot number + Battue/Treibjagd + list of dates.
    """
    blocks = []
    for li, label_lines in enumerate(labels):
        lot_line: Line | None = None
        label_line: Line | None = None
        date_lines: list[Line] = []

        for l in label_lines:
            txt = l.text.strip()
            if is_lot_number(txt) and lot_line is None:
                lot_line = l
            elif is_battue_label(txt) and label_line is None:
                label_line = l
            elif is_date_line(txt):
                date_lines.append(l)

        print(f"[DEBUG] Label {li}: "
              f"lot_line={lot_line.text if lot_line else None}, "
              f"label_line={label_line.text if label_line else None}, "
              f"date_lines={[d.text for d in date_lines]}")

        if lot_line and label_line and date_lines:
            date_lines.sort(key=lambda l: l.cy)
            blocks.append({
                "lot_line": lot_line,
                "label_line": label_line,
                "date_lines": date_lines,
            })
    return blocks


# ---------------------------
# Lot matching heuristics
# ---------------------------

def edit_distance(a: str, b: str) -> int:
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)]
          for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return dp[-1][-1]


def lot_similarity(ocr_lot: str, target_lot: str) -> float:
    a = ocr_lot.strip()
    b = target_lot.strip()
    if not a or not b:
        return 0.0
    dist = edit_distance(a, b)
    return 1 - dist / max(len(a), len(b))


def is_centered(line: Line, img_w: int, tolerance_ratio: float = 0.15) -> bool:
    img_cx = img_w / 2
    dx = abs(line.cx - img_cx)
    return dx <= tolerance_ratio * img_w


def choose_block_for_lot(blocks, target_lot: str, img_w: int):
    """
    Apply heuristics:
    - Prefer exact lot match.
    - Else similar & centered (assume OCR typo).
    - Discard others (neighbor lots).
    """
    exact_matches = []
    typo_candidates = []

    for idx, b in enumerate(blocks):
        ocr_lot = b["lot_line"].text.strip()
        sim = lot_similarity(ocr_lot, target_lot)
        centered = is_centered(b["lot_line"], img_w)
        print(f"[DEBUG] Block {idx}: lot_ocr='{ocr_lot}', sim={sim:.3f}, centered={centered}")

        if ocr_lot == target_lot:
            exact_matches.append(b)
            continue

        if sim >= 0.7 and centered:
            typo_candidates.append((sim, b))

    if exact_matches:
        exact_matches.sort(key=lambda b: abs(b["lot_line"].cx - img_w / 2))
        return exact_matches[0]

    if typo_candidates:
        typo_candidates.sort(key=lambda sb: sb[0], reverse=True)
        return typo_candidates[0][1]

    return None


# ---------------------------
# Date parsing
# ---------------------------

def parse_date_str(s: str) -> _date | None:
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def extract_dates_from_block(block):
    """For now, accept all parsed dates (you can add seasonal filtering later)."""
    dates: list[_date] = []
    for dline in block["date_lines"]:
        dt = parse_date_str(dline.text)
        if not dt:
            print(f"[DEBUG] Could not parse date from '{dline.text}'")
            continue
        print(f"[DEBUG] Accepting date {dt}")
        dates.append(dt)
    return dates


# ---------------------------
# High-level logic
# ---------------------------

def extract_lot_dates_from_output(
    output_texts,
    target_lot: str,
    image: Image.Image,
):
    if isinstance(output_texts, list):
        text = output_texts[0]
    else:
        text = output_texts

    print("[DEBUG] Full OCR string:", repr(text))

    img_w, img_h = image.size
    print(f"[DEBUG] Image size: {img_w}x{img_h}")

    lines = parse_compact_ocr_string(text, img_w, img_h)
    print("[DEBUG] Parsed & denormalized lines (pixel coords):")
    for i, l in enumerate(lines):
        print(f"  {i}: '{l.text}' bbox=({l.x1},{l.y1},{l.x2},{l.y2}) center=({l.cx:.1f},{l.cy:.1f})")

    if not lines:
        print("[DEBUG] No lines parsed at all.")
        return None

    labels = cluster_lines_into_labels(lines, img_w, img_h)
    print(f"[DEBUG] Found {len(labels)} label clusters")
    for li, lab in enumerate(labels):
        print(f"  Label {li}:")
        for l in lab:
            print(f"    '{l.text}' center=({l.cx:.1f},{l.cy:.1f})")

    blocks = build_blocks_from_labels(labels)
    print(f"[DEBUG] Built {len(blocks)} candidate blocks")

    if not blocks:
        print("[DEBUG] No valid blocks (lot+Battue+dates) found.")
        return None

    chosen = choose_block_for_lot(blocks, target_lot, img_w)
    if chosen is None:
        print("[DEBUG] No block matched the target lot according to heuristics.")
        return None

    print("[DEBUG] Chosen block lot:", chosen["lot_line"].text)

    dates = extract_dates_from_block(chosen)
    if not dates:
        print("[DEBUG] Block found but no valid dates.")
        return None

    return {
        "lot_ocr": chosen["lot_line"].text.strip(),
        "lot_centered": is_centered(chosen["lot_line"], img_w),
        "dates": dates,
        "bbox_lot": (
            chosen["lot_line"].x1,
            chosen["lot_line"].y1,
            chosen["lot_line"].x2,
            chosen["lot_line"].y2,
        ),
    }


# ---------------------------
# HunyuanOCR invocation
# ---------------------------

def run_hunyuan_ocr(image_path: str, model_name_or_path: str = "tencent/HunyuanOCR"):
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
    image_inputs = Image.open(image_path)

    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        "Detect and recognize text in the image, "
                        "and output the text coordinates in a formatted manner."
                    ),
                },
            ],
        }
    ]
    messages = [messages1]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
    )

    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16384,
            do_sample=False,
        )

    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        input_ids = inputs.inputs

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = clean_repeated_substrings(
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )
    return image_inputs, output_texts


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--lot", required=True, help="Target lot number, e.g. '110'.")
    parser.add_argument(
        "--model",
        default="tencent/HunyuanOCR",
        help="Model name or path.",
    )
    args = parser.parse_args()

    image, output_texts = run_hunyuan_ocr(args.image, args.model)
    print("Raw OCR output:", output_texts)

    result = extract_lot_dates_from_output(output_texts, args.lot, image)

    if result is None:
        print(f"No valid dates found for lot {args.lot} in this image.")
    else:
        print(f"Chosen OCR lot: {result['lot_ocr']} (centered: {result['lot_centered']})")
        print("Dates:", [d.isoformat() for d in result["dates"]])
        print("Lot bbox (x1,y1,x2,y2):", result["bbox_lot"])


if __name__ == "__main__":
    main()
