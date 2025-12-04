import os
import io
import json
import datetime
import pandas as pd
import requests
import re
import cv2
import numpy as np
from urllib.request import urlopen
from shapely.geometry import Polygon, MultiPolygon
from pyproj import Transformer
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import dateparser
import tempfile   # << kept (no-op, left as requested)

from dataclasses import dataclass
from datetime import datetime as _dt, date as _date

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration

# super-image for upscaling
from super_image import DrlnModel, ImageLoader

# ================================
# --- Global Constants ---
# ================================

BASE_LOTS_FILE = 'hunting_lots.csv'
DATES_FILE = 'hunting_dates.csv'
TILES_DIR = 'tiles'
UPSCALED_TILES_DIR = 'tiles_upscaled'
DEBUG_DIR = 'debug_vlm'
RAW_TEXT_DIR = 'raw_text_output'
API_URL = "https://wms.inspire.geoportail.lu/geoserver/am/ogc/features/v1/collections/AM.HuntingLots/items?f=json&limit=1000&startIndex=0"
WMS_BASE_URL = "https://wmsproxy.geoportail.lu/ogcproxywms"

# --- Hunyuan OCR Configuration ---
HUNYUAN_MODEL_NAME = "tencent/HunyuanOCR"

# --- Super-resolution Configuration ---
SUPERRES_MODEL_NAME = "eugenesiow/drln"  # DRLN model family
SUPERRES_SCALE = 4                       # must match model

# ================================
# --- Utility Functions ---
# ================================

def safe_literal_eval(val):
    """Safely evaluate string representations of lists/tuples."""
    try:
        if isinstance(val, str) and (val.startswith('[') or val.startswith('(')):
            return json.loads(val.replace("'", '"').replace("(", "[").replace(")", "]"))
        return val
    except Exception:
        return []


def ensure_directory(directory: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def file_uptodate(file_path: str, days: int, required_columns: List[str] = None) -> bool:
    """Check if file exists, is recent, and has required columns."""
    if not os.path.exists(file_path):
        return False
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    if (datetime.datetime.now() - mod_time).days > days:
        return False
    if required_columns:
        try:
            df = pd.read_csv(file_path, converters={'dates': safe_literal_eval})
            return all(col in df.columns for col in required_columns)
        except Exception:
            return False
    return True


def load_image_safe(image_path: str) -> Optional[np.ndarray]:
    """Safely load an image with OpenCV, returning None if failed."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# ================================
# --- Part 1: Hunting Lot Data ---
# ================================

def update_lots(url: str) -> pd.DataFrame:
    """Fetches and processes hunting lot geo data."""
    print("Fetching latest hunting lot data from server...")
    try:
        with urlopen(url) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error fetching lot data: {e}")
        return pd.DataFrame()

    features = data.get('features', [])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    processed_lots: List[Dict[str, Any]] = []
    for item in tqdm(features, desc="Processing Lot Geometry"):
        properties = item.get('properties', {})
        lot_num = properties.get('gml_description', 'Unknown')
        geometry = item.get('geometry')

        lot_data = {
            'lot': lot_num,
            'polygon': None,
            'centroid': None,
            'bbox': None
        }

        if geometry:
            geom_type = geometry.get('type')
            coords = geometry.get('coordinates')
            lot_data['polygon'] = coords
            try:
                poly_obj = None
                if geom_type == 'Polygon' and coords:
                    poly_obj = Polygon(coords[0])
                elif geom_type == 'MultiPolygon' and coords:
                    poly_obj = MultiPolygon([Polygon(p[0]) for p in coords if len(p) > 0])

                if poly_obj:
                    centroid = poly_obj.centroid
                    lot_data['centroid'] = (centroid.x, centroid.y)
                    x, y = transformer.transform(centroid.x, centroid.y)
                    lot_data['bbox'] = (x - 1000, y - 1000, x + 1000, y + 1000)
            except Exception as e:
                print(f"Geometry error for lot {lot_num}: {e}")

        processed_lots.append(lot_data)

    df = pd.DataFrame(processed_lots)
    df = df[df['lot'] != 'Unknown'].copy()
    df['lot'] = pd.to_numeric(df['lot'], errors='coerce')
    df = df.dropna(subset=['lot']).astype({'lot': int}).sort_values('lot').reset_index(drop=True)

    df.to_csv(BASE_LOTS_FILE, index=False)
    print(f"Saved full lot geometry data → {BASE_LOTS_FILE}")
    return df

# ================================
# --- Part 2: Tile Download ---
# ================================

def get_tile_path(lot_number: int) -> str:
    """Get path to original tile."""
    return os.path.join(TILES_DIR, f"{lot_number:03d}.png")


def get_upscaled_tile_path(lot_number: int) -> str:
    """Get path to upscaled tile."""
    return os.path.join(UPSCALED_TILES_DIR, f"{lot_number:03d}_x{SUPERRES_SCALE}.png")


def tile_uptodate(lot_number: int, days: int = 7) -> bool:
    """Check if tile is recent."""
    path = get_tile_path(lot_number)
    if not os.path.exists(path):
        return False
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    return (datetime.datetime.now() - mod_time).days <= days


def download_tile(lot_number: int, bounds: Tuple[float, float, float, float]) -> bool:
    """Download WMS tile for a lot."""
    try:
        bbox_str = ",".join([f"{coord:.2f}" for coord in bounds])
        params = {
            'SERVICE': 'WMS', 'VERSION': '1.3.0', 'REQUEST': 'GetMap',
            'FORMAT': 'image/png', 'TRANSPARENT': 'true',
            'LAYERS': 'anf_dates_battues', 'CRS': 'EPSG:3857',
            'STYLES': '', 'WIDTH': '512', 'HEIGHT': '512', 'BBOX': bbox_str
        }
        response = requests.get(WMS_BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        ensure_directory(TILES_DIR)

        with open(get_tile_path(lot_number), 'wb') as f:
            f.write(response.content)

        if os.path.exists(get_tile_path(lot_number)) and os.path.getsize(get_tile_path(lot_number)) > 0:
            return True
        else:
            print(f"Downloaded file is empty or missing for lot {lot_number}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Tile download failed for lot {lot_number}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading tile for lot {lot_number}: {e}")
        return False

# ================================
# --- Super-resolution (DRLN x4) ---
# ================================

class SuperResolutionWrapper:
    """
    Thin wrapper around super-image DRLN x4 model.
    Loaded once and reused for all tiles.
    """
    def __init__(self, model_name: str = SUPERRES_MODEL_NAME, scale: int = SUPERRES_SCALE):
        print(f"Loading super-resolution model '{model_name}' (scale x{scale})...")
        self.model = DrlnModel.from_pretrained(model_name, scale=scale)
        self.scale = scale
        print("Super-resolution model loaded.")

    def _open_with_background(self, input_path: str, bg_color=(255, 255, 255)) -> Image.Image:
        """
        Open a possibly-transparent PNG and composite onto a solid background.
        Default background is white (255,255,255).
        """
        img = Image.open(input_path)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Ensure RGBA
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, bg_color)
            bg.paste(img, mask=img.split()[-1])  # use alpha channel as mask
            return bg
        else:
            # No alpha channel, just convert to RGB
            return img.convert("RGB")

    def upscale_to_file(self, input_path: str, output_path: str) -> bool:
        """
        Upscale an image file and write the result to disk.
        Returns True on success, False on error.
        """
        try:
            # Open with PIL, flatten transparency onto white
            img = self._open_with_background(input_path, bg_color=(255, 255, 255))
            lr = ImageLoader.load_image(img)   # CHW tensor in [0,1]
            sr = self.model(lr)

            ensure_directory(os.path.dirname(output_path))
            # In this super-image version, save_image(tensor, output_file)
            ImageLoader.save_image(sr, output_path)
            return True
        except Exception as e:
            print(f"Super-resolution failed for '{input_path}': {e}")
            return False

# ================================
# --- Hunyuan OCR utilities (from test_hunyuan3.py) ---
# ================================

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

        x1 = int(x1_n * img_w / 1000.0)
        y1 = int(y1_n * img_h / 1000.0)
        x2 = int(x2_n * img_w / 1000.0)
        y2 = int(y2_n * img_h / 1000.0)

        lines.append(Line(text, x1, y1, x2, y2))
    return lines


def lines_are_close(a: Line, b: Line, max_dx: float, max_dy: float) -> bool:
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

    max_dx = img_w * 0.2
    max_dy = img_h * 0.2

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


def parse_date_str(s: str) -> _date | None:
    try:
        return _dt.strptime(s.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def extract_dates_from_block(block):
    dates: list[_date] = []
    for dline in block["date_lines"]:
        dt = parse_date_str(dline.text)
        if not dt:
            print(f"[DEBUG] Could not parse date from '{dline.text}'")
            continue
        print(f"[DEBUG] Accepting date {dt}")
        dates.append(dt)
    return dates


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

# ================================
# --- HunyuanOCR wrapper (reused) ---
# ================================

class HunyuanOCR:
    """
    Lightweight wrapper to load the model/processor once
    and run inference for many tiles.
    """
    def __init__(self, model_name_or_path: str = HUNYUAN_MODEL_NAME):
        print(f"Loading HunyuanOCR model '{model_name_or_path}'...")
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        print("HunyuanOCR model loaded.")

    def run(self, image_path: str):
        """
        Same logic as test_hunyuan3.run_hunyuan_ocr,
        but wrapped for reuse.
        """
        processor = self.processor
        model = self.model

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

# ================================
# --- Main Processing Logic (Hunyuan + Super-res) ---
# ================================

def get_hunt_dates_with_ocr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to extract hunting dates using:
    - WMS tiles (512x512)
    - 4x super-resolution via DRLN
    - HunyuanOCR for OCR
    """
    ensure_directory(TILES_DIR)
    ensure_directory(UPSCALED_TILES_DIR)
    ensure_directory(DEBUG_DIR)
    ensure_directory(RAW_TEXT_DIR)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(RAW_TEXT_DIR, f"hunyuan_superres_log_{timestamp}.txt")
    detailed_log = os.path.join(RAW_TEXT_DIR, f"hunyuan_superres_debug_log_{timestamp}.txt")

    print("\nStep 1: Downloading and preparing tiles...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing Tiles"):
        lot_num = int(row['lot'])

        if row['bbox'] and not tile_uptodate(lot_num):
            success = download_tile(lot_num, row['bbox'])
            if not success:
                print(f"Warning: Failed to download tile for lot {lot_num}")

    # Initialize models once
    try:
        sr_model = SuperResolutionWrapper(SUPERRES_MODEL_NAME, SUPERRES_SCALE)
    except Exception as e:
        print(f"\nFATAL ERROR during super-resolution initialization: {e}")
        exit(1)

    try:
        ocr = HunyuanOCR(HUNYUAN_MODEL_NAME)
    except Exception as e:
        print(f"\nFATAL ERROR during HunyuanOCR initialization: {e}")
        exit(1)

    all_dates: List[List[str]] = []
    stats = {
        'total_lots': len(df),
        'lots_with_dates': 0,
        'failed_lots': 0,
        'no_tile': 0,
        'tile_load_failed': 0,
        'sr_failed': 0,
    }

    print("\nStep 2: Running super-resolution + HunyuanOCR...")
    with open(log_file, 'w', encoding='utf-8') as log, \
         open(detailed_log, 'w', encoding='utf-8') as detail_log:

        log.write("HunyuanOCR + DRLN x4 Super-resolution Log\n" + "="*80 + "\n\n")
        detail_log.write("HunyuanOCR + DRLN x4 Super-resolution Detailed Log\n" + "="*80 + "\n\n")

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting Dates"):
            lot_number = int(row['lot'])
            lot_str = str(lot_number)
            tile_path = get_tile_path(lot_number)
            upscaled_tile_path = get_upscaled_tile_path(lot_number)

            log.write(f"\n--- LOT {lot_number:03d} ---\n")
            detail_log.write(f"\n{'='*80}\nLOT {lot_number:03d}\n{'='*80}\n")

            if not os.path.exists(tile_path):
                log.write("Status: Tile not found.\n")
                detail_log.write("Status: Tile not found.\n")
                all_dates.append([])
                stats['failed_lots'] += 1
                stats['no_tile'] += 1
                continue

            # Light check the original tile with cv2
            test_image = load_image_safe(tile_path)
            if test_image is None:
                log.write("Status: Tile exists but cannot be loaded (cv2 test failed).\n")
                detail_log.write("Status: Tile exists but cannot be loaded (cv2 test failed).\n")
                all_dates.append([])
                stats['failed_lots'] += 1
                stats['tile_load_failed'] += 1
                continue

            # Super-res (with caching)
            if not os.path.exists(upscaled_tile_path):
                detail_log.write(f"Upscaling tile using DRLN x{SUPERRES_SCALE}...\n")
                ok = sr_model.upscale_to_file(tile_path, upscaled_tile_path)
                if not ok:
                    detail_log.write("Super-resolution failed; falling back to original tile.\n")
                    stats['sr_failed'] += 1
                    ocr_input_path = tile_path
                else:
                    ocr_input_path = upscaled_tile_path
            else:
                detail_log.write(f"Using cached upscaled tile: {upscaled_tile_path}\n")
                ocr_input_path = upscaled_tile_path

            try:
                image_pil, output_texts = ocr.run(ocr_input_path)
            except Exception as e:
                msg = f"HunyuanOCR inference error: {e}"
                log.write(msg + "\n")
                detail_log.write(msg + "\n")
                all_dates.append([])
                stats['failed_lots'] += 1
                continue

            detail_log.write("Raw OCR output:\n")
            detail_log.write(repr(output_texts) + "\n\n")

            result = extract_lot_dates_from_output(output_texts, lot_str, image_pil)

            if result is None:
                log.write("Status: No valid dates found for this lot.\n")
                detail_log.write("Status: No valid dates found for this lot.\n")
                all_dates.append([])
                stats['failed_lots'] += 1
                continue

            chosen_lot = result['lot_ocr']
            dates_objs: List[_date] = result['dates']
            dates_strs = [d.strftime("%d/%m/%Y") for d in dates_objs]

            log.write("Status: Dates found.\n")
            log.write(f"OCR lot: {chosen_lot} (target: {lot_str}, centered: {result['lot_centered']})\n")
            log.write(f"Dates: {dates_strs}\n")

            detail_log.write(f"Chosen OCR lot: {chosen_lot} (centered: {result['lot_centered']})\n")
            detail_log.write(f"Dates: {dates_strs}\n")
            detail_log.write(f"Lot bbox (x1,y1,x2,y2): {result['bbox_lot']}\n")

            all_dates.append(dates_strs)
            stats['lots_with_dates'] += 1

            # Save successful images (original; leave behavior as-is)
            success_image = load_image_safe(tile_path)
            if success_image is not None:
                cv2.imwrite(
                    os.path.join(DEBUG_DIR, f"{lot_number:03d}_success.png"),
                    success_image
                )

    print(f"\nLogs saved:")
    print(f"  Summary: {log_file}")
    print(f"  Detailed: {detailed_log}")
    if os.path.exists(DEBUG_DIR):
        print(f"  Success images: {DEBUG_DIR}")

    print("\n=== HunyuanOCR + Super-resolution Summary ===")
    print(f"Total lots processed: {stats['total_lots']}")
    print(f"Lots with dates found: {stats['lots_with_dates']}")
    print(f"Failed (no tile): {stats['no_tile']}")
    print(f"Failed (tile load error): {stats['tile_load_failed']}")
    print(f"Failed (super-resolution errors): {stats['sr_failed']}")
    print(f"Failed (other): {stats['failed_lots'] - stats['no_tile'] - stats['tile_load_failed']}")
    success_rate = (stats['lots_with_dates'] / stats['total_lots']) * 100 if stats['total_lots'] > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")

    df['dates'] = all_dates
    return df

# ================================
# --- Main Execution ---
# ================================

if __name__ == "__main__":
    REQUIRED_COLUMNS = ['lot', 'polygon', 'centroid', 'bbox', 'dates']

    # --- Part 1: Get Lot Data ---
    if file_uptodate(BASE_LOTS_FILE, days=30, required_columns=['lot', 'polygon', 'centroid', 'bbox']):
        print(f"Using recent lot data from '{BASE_LOTS_FILE}'")
        df_lots = pd.read_csv(
            BASE_LOTS_FILE,
            converters={
                'polygon': safe_literal_eval,
                'centroid': safe_literal_eval,
                'bbox': safe_literal_eval
            }
        )
    else:
        print("Lot data is outdated or missing. Fetching new data...")
        df_lots = update_lots(API_URL)
        if df_lots.empty:
            print("Failed to get lot data. Exiting.")
            exit(1)

    # --- Part 2: Get Hunt Dates ---
    if file_uptodate(DATES_FILE, days=1, required_columns=REQUIRED_COLUMNS):
        print(f"\nUsing recent hunting dates from '{DATES_FILE}'")
        df_dates = pd.read_csv(
            DATES_FILE,
            converters={
                'dates': safe_literal_eval,
                'polygon': safe_literal_eval,
                'centroid': safe_literal_eval,
                'bbox': safe_literal_eval
            }
        )
    else:
        print("\nHunting dates file is outdated or missing. Running HunyuanOCR + super-resolution process...")
        df_dates = get_hunt_dates_with_ocr(df_lots.copy())

        df_save = df_dates.copy()
        df_save['dates'] = df_save['dates'].apply(lambda d: tuple(d) if isinstance(d, list) else ())
        df_save[REQUIRED_COLUMNS].to_csv(DATES_FILE, index=False)
        print(f"\nSaved latest hunting dates to '{DATES_FILE}'")

    # --- Part 3: Display Results ---
    print("\n--- Final Results ---")

    df_dates['has_dates'] = df_dates['dates'].apply(lambda d: isinstance(d, (list, tuple)) and len(d) > 0)
    lots_with_dates = df_dates[df_dates['has_dates']].copy()

    print(f"Found dates for {len(lots_with_dates)} / {len(df_dates)} lots.")

    if not lots_with_dates.empty:
        print("\nFirst 15 lots with dates found:")
        for _, row in lots_with_dates.head(15).iterrows():
            dates = row['dates'] if isinstance(row['dates'], list) else list(row['dates'])
            print(f"Lot {row['lot']:03d}: {dates}")
    else:
        print("\nNo hunting dates were found for any lots.")
        print("\nTroubleshooting:")
        print("1. Check that tiles are downloaded in tiles/ and upscaled in tiles_upscaled/")
        print("2. Review detailed log for HunyuanOCR output quality")
        print("3. Check GPU memory usage for DRLN and HunyuanOCR")
