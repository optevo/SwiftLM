#!/usr/bin/env python3
"""
Qualitative VLM comparison — sends open-ended descriptive prompts to a running
SwiftLM server and prints the full model responses.

Designed to be run once per model; results captured and compared offline.

Usage:
  python3 tests/vision/qual_compare.py <model_label> <base_url> [--timeout N]

Example:
  python3 tests/vision/qual_compare.py "Qwen3-VL-4B" http://127.0.0.1:18001
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import textwrap
import time
import urllib.request
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# Glyph font (5×7 bitmap, shared with run_model_tests.py)
# ---------------------------------------------------------------------------

_GLYPHS: dict[str, list[int]] = {
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "S": [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "4": [0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010, 0b00010],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110],
}


def _draw_char(img: np.ndarray, ch: str, x0: int, y0: int, scale: int = 8) -> None:
    glyph = _GLYPHS.get(ch.upper())
    if glyph is None:
        return
    for row_i, bits in enumerate(glyph):
        for col_i in range(5):
            if bits & (1 << (4 - col_i)):
                y = y0 + row_i * scale
                x = x0 + col_i * scale
                if 0 <= y < img.shape[0] - scale and 0 <= x < img.shape[1] - scale:
                    img[y:y + scale, x:x + scale] = 0


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

def _encode_png(arr: np.ndarray) -> bytes:
    h, w = arr.shape[:2]
    def chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    sig  = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw  = b"".join(b"\x00" + bytes(arr[y].tobytes()) for y in range(h))
    idat = chunk(b"IDAT", zlib.compress(raw, 9))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def make_chart_image() -> str:
    """320×224 four-bar chart — red, blue, green (tallest), yellow."""
    W, H = 320, 224
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    img[180:182, 30:290] = 0
    img[20:182,  30:32]  = 0
    for x, h_px, col in [
        ( 50, 100, (220,  50,  50)),
        (110,  60, ( 50, 150, 220)),
        (170, 140, ( 50, 200,  80)),
        (230,  80, (220, 180,  30)),
    ]:
        img[180 - h_px:180, x:x + 40] = col
    return b64(_encode_png(img))


def make_labeled_chart_image() -> str:
    """
    320×260 bar chart with A/B/C/D labels on x-axis.
    A=60 red, B=100 blue, C=140 green (tallest), D=80 yellow.
    """
    W, H = 320, 260
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    img[180:182, 30:290] = 0
    img[20:182,  30:32]  = 0
    bars = [
        ( 50,  60, (220,  50,  50), "A"),
        (110, 100, ( 50, 150, 220), "B"),
        (170, 140, ( 50, 200,  80), "C"),
        (230,  80, (220, 180,  30), "D"),
    ]
    for x, h, col, lbl in bars:
        img[180 - h:180, x:x + 40] = col
        _draw_char(img, lbl, x + 8, 190, scale=5)
    return b64(_encode_png(img))


def make_grid_image() -> str:
    """320×224 grid of 6 coloured squares (3 columns × 2 rows)."""
    img = np.ones((224, 320, 3), dtype=np.uint8) * 220
    colours = [
        (220, 50,  50),   # red
        (50,  180, 50),   # green
        (50,  100, 220),  # blue
        (220, 140, 40),   # orange
        (140, 50,  180),  # purple
        (220, 200, 40),   # yellow
    ]
    positions = [(40, 20), (130, 20), (220, 20), (40, 120), (130, 120), (220, 120)]
    for (x, y), col in zip(positions, colours):
        img[y:y+80, x:x+80] = col
    return b64(_encode_png(img))


def make_spatial_image() -> str:
    """320×224 image: red square top-left, blue square bottom-right."""
    img = np.ones((224, 320, 3), dtype=np.uint8) * 245
    img[30:90,   200:260] = (220, 50, 50)    # red — top-right
    img[140:200,  60:120] = (50, 100, 220)   # blue — bottom-left
    return b64(_encode_png(img))


def make_three_shape_image() -> str:
    """
    320×240 image: red square top-left, green square centre, blue square bottom-right.
    Tests multi-object spatial reasoning.
    """
    img = np.ones((240, 320, 3), dtype=np.uint8) * 245
    img[20:80,   20:80]   = (220, 50, 50)    # red — top-left
    img[90:160, 125:195]  = (50, 200, 80)    # green — centre
    img[165:225, 245:305] = (50, 80, 220)    # blue — bottom-right
    return b64(_encode_png(img))


def make_text_image() -> str:
    """224×224 white image with text 'HELLO 42' rendered via pixel font."""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    text = "HELLO 42"
    scale = 6
    x_start = 10
    y_start = 80
    for i, ch in enumerate(text):
        _draw_char(img, ch, x_start + i * (5 * scale + scale), y_start, scale)
    return b64(_encode_png(img))


def make_multiline_text_image() -> str:
    """
    White image with three lines: HELLO / STOP / 42.
    Tests multi-line OCR capability.
    """
    lines = ["HELLO", "STOP", "42"]
    scale = 12
    char_w = 5 * scale + scale
    line_h = 7 * scale + scale
    W = 5 * char_w + 40
    H = len(lines) * line_h + 40
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    for line_i, line in enumerate(lines):
        y0 = 20 + line_i * line_h
        x0 = 20
        for ch in line.upper():
            _draw_char(img, ch, x0, y0, scale)
            x0 += char_w
    return b64(_encode_png(img))


def make_counting_image(n: int = 7) -> str:
    """White image with n black dots arranged in a loose grid."""
    import random
    random.seed(42)
    img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    placed: list[tuple[int, int]] = []
    attempts = 0
    while len(placed) < n and attempts < 500:
        cx = random.randint(20, 200)
        cy = random.randint(20, 200)
        if all(abs(cx - px) > 30 or abs(cy - py) > 30 for px, py in placed):
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    if dx*dx + dy*dy <= 100:
                        img[cy+dy, cx+dx] = (0, 0, 0)
            placed.append((cx, cy))
        attempts += 1
    return b64(_encode_png(img))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

PROBES = [
    {
        "id":     "chart-describe",
        "label":  "Chart description",
        "image":  make_chart_image,
        "prompt": (
            "Describe what you see in this image. "
            "How many bars are there, what colours are they, and which is the tallest?"
        ),
        "max_tokens": 120,
    },
    {
        "id":     "chart-labelled",
        "label":  "Labelled chart reading",
        "image":  make_labeled_chart_image,
        "prompt": (
            "This bar chart has bars labelled A, B, C, D along the x-axis. "
            "Which label has the tallest bar? What colour is each bar? "
            "Describe the relative heights."
        ),
        "max_tokens": 150,
    },
    {
        "id":     "grid-colours",
        "label":  "Colour grid",
        "image":  make_grid_image,
        "prompt": (
            "Describe all the coloured squares you see in this image. "
            "How many are there, what colours, and how are they arranged?"
        ),
        "max_tokens": 150,
    },
    {
        "id":     "spatial-describe",
        "label":  "Spatial reasoning (2 objects)",
        "image":  make_spatial_image,
        "prompt": (
            "Describe what you see. Where is the red shape relative to the blue shape? "
            "Be specific about their positions in the image."
        ),
        "max_tokens": 100,
    },
    {
        "id":     "spatial-three",
        "label":  "Spatial reasoning (3 objects)",
        "image":  make_three_shape_image,
        "prompt": (
            "There are three coloured squares in this image. "
            "Describe the colour and position of each one. "
            "Which colour is in the centre?"
        ),
        "max_tokens": 120,
    },
    {
        "id":     "text-ocr",
        "label":  "OCR / text reading",
        "image":  make_text_image,
        "prompt": "Read and transcribe any text or numbers visible in this image.",
        "max_tokens": 40,
    },
    {
        "id":     "multiline-ocr",
        "label":  "Multi-line OCR",
        "image":  make_multiline_text_image,
        "prompt": (
            "Read all lines of text in this image. "
            "List each line separately in the order they appear top to bottom."
        ),
        "max_tokens": 60,
    },
    {
        "id":     "counting-describe",
        "label":  "Counting + description",
        "image":  lambda: make_counting_image(7),
        "prompt": (
            "How many dots or circles are in this image? "
            "Describe their arrangement briefly."
        ),
        "max_tokens": 80,
    },
    {
        "id":     "scene-type",
        "label":  "Image type / scene",
        "image":  make_chart_image,
        "prompt": (
            "What type of image is this — a chart/graph, a photograph, a document, "
            "or something else? Describe what it shows in one or two sentences."
        ),
        "max_tokens": 80,
    },
]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(base_url: str, b64_img: str, prompt: str, max_tokens: int, timeout: int) -> tuple[str, float]:
    payload = json.dumps({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        elapsed = time.monotonic() - t0
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip(), elapsed
    except Exception as e:
        return f"[ERROR: {e}]", time.monotonic() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 80, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_label", help="Display name for the model (e.g. 'Qwen3-VL-4B')")
    p.add_argument("base_url",    help="Server base URL, e.g. http://127.0.0.1:18001")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    SEP  = "=" * 72
    SEP2 = "-" * 72

    print(f"\n{SEP}")
    print(f"  MODEL: {args.model_label}")
    print(f"  URL:   {args.base_url}")
    print(SEP)

    for probe in PROBES:
        img = probe["image"]()
        print(f"\n  [{probe['label']}]")
        print(f"  Q: {probe['prompt']}")
        print(SEP2)
        answer, elapsed = query(
            args.base_url, img, probe["prompt"], probe["max_tokens"], args.timeout
        )
        print(wrap(answer))
        print(f"  ({elapsed:.1f}s)")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
