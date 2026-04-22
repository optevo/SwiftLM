#!/usr/bin/env python3
"""
VLM capability benchmark for SwiftLM.

Evaluates vision-language models across six capability dimensions using
synthetically generated images (no network dependency) plus optional
real-world fixture images downloaded from stable Wikimedia Commons URLs.

  1. OCR              — text and digit reading (single word, multi-line)
  2. Chart analysis   — bar chart comprehension, labelled-axis charts
  3. Counting         — object counting at varying densities
  4. Colour           — colour naming and presence detection
  5. Spatial reasoning— positional relationships between objects
  6. Scene classification — image type identification

Each test dimension reports accuracy (N/M) and average inference latency (ms).
Tests are model-appropriate — small/specialist models skip dimensions they are
not designed for.

Usage:
  python3 tests/vision/run_model_tests.py <model_name> <base_url>
          [--timeout N]  [--pass-threshold 0.6]

  model_name:      directory name, used to select the appropriate suite
  base_url:        server base URL, e.g. http://127.0.0.1:18001
  --timeout:       per-request timeout in seconds (default 90)
  --pass-threshold: min fraction of tests that must pass to exit 0 (default 0.6)

Exit: 0 if overall score ≥ threshold, else 1.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# Image generation — pure Python + numpy, no external dependencies
# ---------------------------------------------------------------------------

def _encode_png(arr: "np.ndarray") -> bytes:
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


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# 5×7 bitmap glyph font — each row is a 5-bit bitmask (MSB = leftmost pixel)
# ---------------------------------------------------------------------------

_GLYPHS: dict[str, list[int]] = {
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10010, 0b10001, 0b10001],
    "S": [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
    "3": [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    "4": [0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110],
    "6": [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110],
}


def _draw_char(img: "np.ndarray", ch: str, x0: int, y0: int, scale: int = 8) -> None:
    """Draw a single character glyph onto img in black at (x0, y0)."""
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
# Synthetic image generators
# ---------------------------------------------------------------------------

# --- Charts ---

def make_chart_image() -> str:
    """
    320×224 bar chart with axes and four coloured bars:
      red (h=100), blue (h=60), green (h=140, tallest), yellow (h=80).
    """
    import numpy as np
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
    return _b64(_encode_png(img))


def make_labeled_chart_image() -> str:
    """
    320×260 bar chart with A/B/C/D labels under the x-axis.
    Bar heights: A=60 (red), B=100 (blue), C=140 (green, tallest), D=80 (yellow).
    """
    import numpy as np
    W, H = 320, 260
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    img[180:182, 30:290] = 0   # x-axis
    img[20:182,  30:32]  = 0   # y-axis
    bars = [
        ( 50,  60, (220,  50,  50), "A"),
        (110, 100, ( 50, 150, 220), "B"),
        (170, 140, ( 50, 200,  80), "C"),   # tallest, green
        (230,  80, (220, 180,  30), "D"),
    ]
    for x, h, col, lbl in bars:
        img[180 - h:180, x:x + 40] = col
        # Draw label centred under bar (glyph is 5×7 at scale=5 → 25×35 px)
        _draw_char(img, lbl, x + 8, 190, scale=5)
    return _b64(_encode_png(img))


# --- Counting ---

def make_grid_image() -> str:
    """
    320×224 image with 6 coloured squares (3×2 grid) on a light-grey background:
    red, green, blue (top row) / orange, purple, yellow (bottom).
    """
    import numpy as np
    img = np.ones((224, 320, 3), dtype=np.uint8) * 220
    for color, x, y in [
        ((255,   0,   0),  10,  10),
        ((  0, 200,   0), 115,  10),
        ((  0,   0, 255), 220,  10),
        ((255, 165,   0),  10, 120),
        ((128,   0, 128), 115, 120),
        ((255, 200,   0), 220, 120),
    ]:
        img[y:y + 90, x:x + 90] = color
    return _b64(_encode_png(img))


def make_dots_image(n: int) -> str:
    """
    320×224 white image containing N large black filled circles in a row.
    Used for simple object counting (n ≤ 8).
    """
    import numpy as np
    img = np.ones((224, 320, 3), dtype=np.uint8) * 255
    r = 18
    spacing = 320 // (n + 1)
    cy = 112
    for i in range(n):
        cx = spacing * (i + 1)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < 224 and 0 <= px < 320:
                        img[py, px] = 0
    return _b64(_encode_png(img))


# --- OCR ---

def make_text_image(text: str, scale: int = 16) -> str:
    """Render TEXT using a 5×7 bitmap font; white background, black text."""
    import numpy as np
    gap = scale
    W = len(text) * (5 * scale + gap) - gap + 2 * 20
    H = 7 * scale + 2 * 20
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    x0 = 20
    for ch in text.upper():
        _draw_char(img, ch, x0, 20, scale)
        x0 += 5 * scale + gap
    return _b64(_encode_png(img))


def make_multiline_text_image(lines: list[str], scale: int = 12) -> str:
    """
    Render multiple lines of text using the 5×7 bitmap font.
    Lines are separated by one glyph-height of whitespace.
    """
    import numpy as np
    char_w = 5 * scale + scale    # character stride (5 px wide + gap)
    line_h = 7 * scale + scale    # line stride (7 px tall + gap)
    max_chars = max(len(l) for l in lines) if lines else 1
    W = max(max_chars * char_w + 40, 200)
    H = len(lines) * line_h + 40
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    for line_i, line in enumerate(lines):
        y0 = 20 + line_i * line_h
        x0 = 20
        for ch in line.upper():
            _draw_char(img, ch, x0, y0, scale)
            x0 += char_w
    return _b64(_encode_png(img))


# --- Spatial reasoning ---

def make_spatial_image() -> str:
    """
    320×224 image: red square in top-left quadrant, blue square in bottom-right.
    Correct answers: red is above blue; blue is to the right of red.
    """
    import numpy as np
    img = np.ones((224, 320, 3), dtype=np.uint8) * 240
    img[ 20:100,  20:100] = (220, 50, 50)   # red — top-left
    img[130:210, 230:310] = ( 50, 80, 220)  # blue — bottom-right
    return _b64(_encode_png(img))


def make_three_shape_image() -> str:
    """
    320×240 image: red circle top-left, blue square bottom-right, green rectangle centre.
    Tests multi-object spatial reasoning.
    """
    import numpy as np
    img = np.ones((240, 320, 3), dtype=np.uint8) * 245
    # Red square — top-left
    img[20:80, 20:80] = (220, 50, 50)
    # Green square — centre
    img[90:160, 125:195] = (50, 200, 80)
    # Blue square — bottom-right
    img[165:225, 245:305] = (50, 80, 220)
    return _b64(_encode_png(img))


# --- Colour ---

def make_solid_image(r: int, g: int, b: int, size: int = 224) -> str:
    """Solid-colour square image (size×size)."""
    import numpy as np
    img = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
    return _b64(_encode_png(img))


# ---------------------------------------------------------------------------
# Real-world fixture support
# Downloads stable Wikimedia Commons images on first run; cached locally.
# Tests using fixtures are added to suites only when the image is available.
# ---------------------------------------------------------------------------

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Stable Wikimedia Commons images (CC BY-SA or PD) used as real-world test material
_FIXTURE_URLS: dict[str, tuple[str, str]] = {
    # A pair of simple bar graphs — good for chart classification on a real image
    "wikimedia-bar-chart": (
        "https://upload.wikimedia.org/wikipedia/commons/2/28/Bar_graphs.png",
        "wikimedia_bar_charts.png",
    ),
    # A stop sign photograph — OCR of real-world text in a photographic context
    "wikimedia-stop-sign": (
        "https://upload.wikimedia.org/wikipedia/commons/8/81/Stop_sign.png",
        "wikimedia_stop_sign.png",
    ),
}


def _try_fixture(key: str) -> str | None:
    """
    Return a base64-encoded fixture image, downloading it if not yet cached.
    Returns None silently if the download fails (allows offline test runs).
    """
    entry = _FIXTURE_URLS.get(key)
    if entry is None:
        return None
    url, filename = entry
    path = os.path.join(_FIXTURE_DIR, filename)
    if not os.path.exists(path):
        try:
            os.makedirs(_FIXTURE_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        except Exception:
            return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTTP + timing
# ---------------------------------------------------------------------------

def ask(base_url: str, b64_img: str, prompt: str, max_tokens: int,
        timeout: int) -> tuple[str, float]:
    """
    Send a vision completion request.  Returns (response_text, elapsed_ms).
    Raises on HTTP/network error.
    """
    payload = json.dumps({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return data["choices"][0]["message"]["content"], elapsed_ms


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_contains(label: str, response: str, keyword: str) -> None:
    if keyword.lower() not in response.lower():
        raise AssertionError(
            f"expected '{keyword}' in response — got: {response[:200]}"
        )


def assert_number_in(label: str, response: str, expected: str) -> None:
    if not re.search(rf"\b{re.escape(expected)}\b", response):
        raise AssertionError(
            f"expected number '{expected}' in response — got: {response[:200]}"
        )


def assert_colour_count(label: str, response: str, colours: list[str],
                        min_match: int) -> None:
    found = [c for c in colours if c.lower() in response.lower()]
    if len(found) < min_match:
        raise AssertionError(
            f"expected ≥{min_match} of {colours}; found {found} — got: {response[:200]}"
        )


def assert_all_contain(label: str, response: str, keywords: list[str]) -> None:
    """Assert every keyword appears (case-insensitive) in the response."""
    missing = [k for k in keywords if k.lower() not in response.lower()]
    if missing:
        raise AssertionError(
            f"missing {missing} in response — got: {response[:200]}"
        )


# ---------------------------------------------------------------------------
# Test case and dimension data structures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    label: str
    image_fn: object        # callable → str (base64)
    prompt: str
    max_tokens: int
    assertion: object       # callable(str) → None | raises AssertionError
    dimension: str


@dataclass
class DimResult:
    passed: int = 0
    total: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def avg_latency(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


# ---------------------------------------------------------------------------
# Per-model test suites
# ---------------------------------------------------------------------------

def build_fastvlm_suite() -> list[TestCase]:
    """
    FastVLM-0.5B — Apple's lightweight visual routing model.
    Focus: scene classification (its primary use case), colour detection, basic
    counting, chart recognition. Tests include both synthetic and Wikimedia fixtures.
    Skip: OCR, complex spatial (model is not trained for these).
    """
    chart_img  = make_chart_image()
    grid_img   = make_grid_image()
    red_img    = make_solid_image(220, 50, 50)
    dots4_img  = make_dots_image(4)
    stop_img   = make_text_image("STOP")

    tests: list[TestCase] = [
        # Colour
        TestCase("colour-naming", lambda: red_img,
                 "What colour is this image? Reply with just the colour name.",
                 8, lambda r: assert_contains("colour-naming", r, "red"),
                 "Colour"),
        TestCase("colour-presence", lambda: grid_img,
                 "Does this image contain red, green, and blue squares? Answer yes or no.",
                 5, lambda r: assert_contains("colour-presence", r, "yes"),
                 "Colour"),
        # Chart
        TestCase("chart-classification", lambda: chart_img,
                 "Is this image a bar chart? Answer yes or no.",
                 5, lambda r: assert_contains("chart-classification", r, "yes"),
                 "Chart"),
        TestCase("chart-bar-count", lambda: chart_img,
                 "How many bars are in this bar chart? Reply with just the number.",
                 5, lambda r: assert_number_in("chart-bar-count", r, "4"),
                 "Chart"),
        # Counting
        TestCase("counting-squares", lambda: grid_img,
                 "How many coloured squares are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-squares", r, "6"),
                 "Counting"),
        TestCase("counting-dots", lambda: dots4_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-dots", r, "4"),
                 "Counting"),
        # Scene classification — FastVLM's actual design purpose
        TestCase("scene-chart-vs-photo", lambda: chart_img,
                 "Is this a chart/graph or a real-world photograph? Reply: 'chart' or 'photo'.",
                 5, lambda r: assert_contains("scene-chart-vs-photo", r, "chart"),
                 "Scene"),
        TestCase("scene-text-vs-chart", lambda: stop_img,
                 "Is this image mostly text, or is it a bar chart? Reply: 'text' or 'chart'.",
                 5, lambda r: assert_contains("scene-text-vs-chart", r, "text"),
                 "Scene"),
    ]

    # Real-world fixture: Wikimedia bar chart
    real_chart = _try_fixture("wikimedia-bar-chart")
    if real_chart is not None:
        _rc = real_chart
        tests.append(TestCase(
            "real-chart-type", lambda: _rc,
            "Is this image a bar chart or a real-world photograph? Reply: 'chart' or 'photo'.",
            5, lambda r: assert_contains("real-chart-type", r, "chart"),
            "Chart",
        ))

    return tests


def build_olmocr_suite() -> list[TestCase]:
    """
    olmOCR-2-7B — Allen AI document OCR model.
    Focus: text/digit reading (primary task), including multi-line OCR.
    Includes a real-world photograph fixture test for the stop sign.
    """
    stop_img  = make_text_image("STOP")
    n42_img   = make_text_image("42")
    n7_img    = make_text_image("7")
    ml_img    = make_multiline_text_image(["HELLO", "STOP", "42"])
    chart_img = make_chart_image()
    grid_img  = make_grid_image()
    red_img   = make_solid_image(220, 50, 50)
    dots3_img = make_dots_image(3)

    tests: list[TestCase] = [
        # OCR — olmOCR's primary capability
        TestCase("ocr-word", lambda: stop_img,
                 "What text is written in this image? Reply with just the word.",
                 10, lambda r: assert_contains("ocr-word", r, "STOP"),
                 "OCR"),
        TestCase("ocr-number-2digit", lambda: n42_img,
                 "What number is shown in this image? Reply with just the number.",
                 5, lambda r: assert_contains("ocr-number-2digit", r, "42"),
                 "OCR"),
        TestCase("ocr-number-1digit", lambda: n7_img,
                 "What single digit is shown in this image? Reply with just the digit.",
                 5, lambda r: assert_contains("ocr-number-1digit", r, "7"),
                 "OCR"),
        TestCase("ocr-multiline", lambda: ml_img,
                 "Read all lines of text in this image. List each line separately.",
                 50, lambda r: assert_all_contain("ocr-multiline", r, ["HELLO", "STOP"]),
                 "OCR"),
        # Chart
        TestCase("chart-classification", lambda: chart_img,
                 "Is this image a bar chart with coloured bars? Answer yes or no.",
                 5, lambda r: assert_contains("chart-classification", r, "yes"),
                 "Chart"),
        TestCase("chart-bar-count", lambda: chart_img,
                 "How many bars are in this bar chart? Reply with just the number.",
                 5, lambda r: assert_number_in("chart-bar-count", r, "4"),
                 "Chart"),
        # Colour
        TestCase("colour-naming", lambda: red_img,
                 "What colour is this image? Reply with just the colour name.",
                 8, lambda r: assert_contains("colour-naming", r, "red"),
                 "Colour"),
        # Counting
        TestCase("counting-squares", lambda: grid_img,
                 "How many coloured squares are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-squares", r, "6"),
                 "Counting"),
        TestCase("counting-dots", lambda: dots3_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-dots", r, "3"),
                 "Counting"),
    ]

    # Real-world OCR: stop sign in a photographic context (harder than synthetic)
    real_stop = _try_fixture("wikimedia-stop-sign")
    if real_stop is not None:
        _rs = real_stop
        tests.append(TestCase(
            "real-ocr-sign", lambda: _rs,
            "What word is written on the sign in this image? Reply with just the word.",
            10, lambda r: assert_contains("real-ocr-sign", r, "STOP"),
            "OCR",
        ))

    return tests


def build_qwen25vl_suite() -> list[TestCase]:
    """
    Qwen2.5-VL-3B-Instruct-6bit — general VLM (3B, 6-bit).
    Full suite: OCR, chart, counting, colour, spatial, scene.
    Includes harder labelled-chart test and multi-line OCR.
    """
    chart_img     = make_chart_image()
    labeled_chart = make_labeled_chart_image()
    grid_img      = make_grid_image()
    stop_img      = make_text_image("STOP")
    n42_img       = make_text_image("42")
    n7_img        = make_text_image("7")
    ml_img        = make_multiline_text_image(["HELLO", "STOP", "42"])
    spatial_img   = make_spatial_image()
    red_img       = make_solid_image(220, 50, 50)
    dots5_img     = make_dots_image(5)

    return [
        # OCR
        TestCase("ocr-word", lambda: stop_img,
                 "What text is written in this image? Reply with just the word.",
                 10, lambda r: assert_contains("ocr-word", r, "STOP"),
                 "OCR"),
        TestCase("ocr-2digit", lambda: n42_img,
                 "What number is shown in this image? Reply with just the number.",
                 5, lambda r: assert_contains("ocr-2digit", r, "42"),
                 "OCR"),
        TestCase("ocr-1digit", lambda: n7_img,
                 "What single digit is shown? Reply with just the digit.",
                 5, lambda r: assert_contains("ocr-1digit", r, "7"),
                 "OCR"),
        TestCase("ocr-multiline", lambda: ml_img,
                 "Read each line of text in this image. List the lines in order.",
                 50, lambda r: assert_all_contain("ocr-multiline", r, ["HELLO", "STOP"]),
                 "OCR"),
        # Chart
        TestCase("chart-tallest-colour", lambda: chart_img,
                 "What colour is the tallest bar in this chart? Reply with just the colour name.",
                 10, lambda r: assert_contains("chart-tallest-colour", r, "green"),
                 "Chart"),
        TestCase("chart-classification", lambda: chart_img,
                 "Is this a bar chart? Answer yes or no.",
                 5, lambda r: assert_contains("chart-classification", r, "yes"),
                 "Chart"),
        TestCase("chart-labelled-tallest", lambda: labeled_chart,
                 "This bar chart has bars labelled A, B, C, D under the x-axis. "
                 "Which label is the tallest bar? Reply with just the letter.",
                 5, lambda r: assert_contains("chart-labelled-tallest", r, "C"),
                 "Chart"),
        # Counting
        TestCase("counting-squares", lambda: grid_img,
                 "How many coloured squares are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-squares", r, "6"),
                 "Counting"),
        TestCase("counting-dots", lambda: dots5_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-dots", r, "5"),
                 "Counting"),
        # Colour
        TestCase("colour-naming", lambda: red_img,
                 "What colour is this solid image? Reply with just the colour name.",
                 8, lambda r: assert_contains("colour-naming", r, "red"),
                 "Colour"),
        TestCase("colour-list", lambda: grid_img,
                 "List the distinct colours of the squares in this image.",
                 60, lambda r: assert_colour_count("colour-list", r,
                     ["red","green","blue","orange","purple","yellow"], min_match=4),
                 "Colour"),
        # Spatial
        TestCase("spatial-above-below", lambda: spatial_img,
                 "Is the red square above or below the blue square? Reply: 'above' or 'below'.",
                 10, lambda r: assert_contains("spatial-above-below", r, "above"),
                 "Spatial"),
        TestCase("spatial-left-right", lambda: spatial_img,
                 "Is the blue square to the left or right of the red square? Reply: 'left' or 'right'.",
                 10, lambda r: assert_contains("spatial-left-right", r, "right"),
                 "Spatial"),
        # Scene
        TestCase("scene-type", lambda: chart_img,
                 "Is this a chart/graph or a real-world photograph? Reply: 'chart' or 'photo'.",
                 5, lambda r: assert_contains("scene-type", r, "chart"),
                 "Scene"),
    ]


def build_qwen3vl_suite() -> list[TestCase]:
    """
    Qwen3-VL (4B and 8B) — general VLM, 4-bit quantised.
    Full suite including OCR and real-world fixture images.
    Note: 4B OCR is unreliable; 8B passes consistently.
    """
    chart_img     = make_chart_image()
    labeled_chart = make_labeled_chart_image()
    grid_img      = make_grid_image()
    stop_img      = make_text_image("STOP")
    n42_img       = make_text_image("42")
    spatial_img   = make_spatial_image()
    red_img       = make_solid_image(220, 50, 50)
    dots5_img     = make_dots_image(5)

    tests: list[TestCase] = [
        # OCR
        TestCase("ocr-word", lambda: stop_img,
                 "What text is written in this image? Reply with just the word.",
                 10, lambda r: assert_contains("ocr-word", r, "STOP"),
                 "OCR"),
        TestCase("ocr-2digit", lambda: n42_img,
                 "What number is shown in this image? Reply with just the number.",
                 5, lambda r: assert_contains("ocr-2digit", r, "42"),
                 "OCR"),
        # Chart
        TestCase("chart-bar-count", lambda: chart_img,
                 "How many bars are in this chart? Reply with just the number.",
                 5, lambda r: assert_number_in("chart-bar-count", r, "4"),
                 "Chart"),
        TestCase("chart-tallest-colour", lambda: chart_img,
                 "What colour is the tallest bar? Reply with just the colour name.",
                 10, lambda r: assert_contains("chart-tallest-colour", r, "green"),
                 "Chart"),
        TestCase("chart-classification", lambda: chart_img,
                 "Is this a bar chart? Answer yes or no.",
                 5, lambda r: assert_contains("chart-classification", r, "yes"),
                 "Chart"),
        TestCase("chart-labelled-tallest", lambda: labeled_chart,
                 "This bar chart has bars labelled A, B, C, D under the x-axis. "
                 "Which label is the tallest bar? Reply with just the letter.",
                 5, lambda r: assert_contains("chart-labelled-tallest", r, "C"),
                 "Chart"),
        # Counting
        TestCase("counting-squares", lambda: grid_img,
                 "How many coloured squares are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-squares", r, "6"),
                 "Counting"),
        TestCase("counting-dots", lambda: dots5_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-dots", r, "5"),
                 "Counting"),
        # Colour
        TestCase("colour-naming", lambda: red_img,
                 "What colour is this image? Reply with just the colour name.",
                 8, lambda r: assert_contains("colour-naming", r, "red"),
                 "Colour"),
        TestCase("colour-list", lambda: grid_img,
                 "List the distinct colours of all the squares in this image.",
                 60, lambda r: assert_colour_count("colour-list", r,
                     ["red","green","blue","orange","purple","yellow"], min_match=4),
                 "Colour"),
        # Spatial
        TestCase("spatial-above-below", lambda: spatial_img,
                 "Is the red square above or below the blue square? Reply: 'above' or 'below'.",
                 10, lambda r: assert_contains("spatial-above-below", r, "above"),
                 "Spatial"),
        TestCase("spatial-left-right", lambda: spatial_img,
                 "Is the blue square to the left or right of the red square? Reply: 'left' or 'right'.",
                 10, lambda r: assert_contains("spatial-left-right", r, "right"),
                 "Spatial"),
        # Scene
        TestCase("scene-type", lambda: chart_img,
                 "Is this a chart/graph or a real-world photograph? Reply: 'chart' or 'photo'.",
                 5, lambda r: assert_contains("scene-type", r, "chart"),
                 "Scene"),
    ]

    # Real-world fixtures
    real_chart = _try_fixture("wikimedia-bar-chart")
    if real_chart is not None:
        _rc = real_chart
        tests.append(TestCase(
            "real-chart-type", lambda: _rc,
            "Is this image a bar chart or a real-world photograph? Reply: 'chart' or 'photo'.",
            5, lambda r: assert_contains("real-chart-type", r, "chart"),
            "Chart",
        ))

    real_stop = _try_fixture("wikimedia-stop-sign")
    if real_stop is not None:
        _rs = real_stop
        tests.append(TestCase(
            "real-ocr-sign", lambda: _rs,
            "What word is written on the sign in this image? Reply with just the word.",
            10, lambda r: assert_contains("real-ocr-sign", r, "STOP"),
            "OCR",
        ))

    return tests


def build_qwen36_suite() -> list[TestCase]:
    """
    Qwen3.6-35B-A3B-VLM-4bit — 35B MoE VLM; full suite including OCR.
    OCR is reliable at this scale despite 4-bit quantisation.
    Includes labelled-chart test and multi-line OCR as harder variants.
    """
    chart_img     = make_chart_image()
    labeled_chart = make_labeled_chart_image()
    grid_img      = make_grid_image()
    spatial_img   = make_spatial_image()
    three_img     = make_three_shape_image()
    red_img       = make_solid_image(220, 50, 50)
    stop_img      = make_text_image("STOP")
    n42_img       = make_text_image("42")
    n7_img        = make_text_image("7")
    ml_img        = make_multiline_text_image(["HELLO", "STOP", "42"])
    dots3_img     = make_dots_image(3)
    dots7_img     = make_dots_image(7)

    tests: list[TestCase] = [
        # OCR
        TestCase("ocr-word", lambda: stop_img,
                 "What text is written in this image? Reply with just the word.",
                 10, lambda r: assert_contains("ocr-word", r, "STOP"),
                 "OCR"),
        TestCase("ocr-2digit", lambda: n42_img,
                 "What number is shown in this image? Reply with just the number.",
                 5, lambda r: assert_contains("ocr-2digit", r, "42"),
                 "OCR"),
        TestCase("ocr-1digit", lambda: n7_img,
                 "What single digit is shown? Reply with just the digit.",
                 5, lambda r: assert_contains("ocr-1digit", r, "7"),
                 "OCR"),
        TestCase("ocr-multiline", lambda: ml_img,
                 "Read each line of text in this image. List the lines in order.",
                 50, lambda r: assert_all_contain("ocr-multiline", r, ["HELLO", "STOP"]),
                 "OCR"),
        # Chart
        TestCase("chart-bar-count", lambda: chart_img,
                 "How many bars are in this chart? Reply with just the number.",
                 5, lambda r: assert_number_in("chart-bar-count", r, "4"),
                 "Chart"),
        TestCase("chart-tallest-colour", lambda: chart_img,
                 "What colour is the tallest bar? Reply with just the colour name.",
                 10, lambda r: assert_contains("chart-tallest-colour", r, "green"),
                 "Chart"),
        TestCase("chart-labelled-tallest", lambda: labeled_chart,
                 "This bar chart has bars labelled A, B, C, D under the x-axis. "
                 "Which label is the tallest bar? Reply with just the letter.",
                 5, lambda r: assert_contains("chart-labelled-tallest", r, "C"),
                 "Chart"),
        # Counting
        TestCase("counting-3dots", lambda: dots3_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-3dots", r, "3"),
                 "Counting"),
        TestCase("counting-7dots", lambda: dots7_img,
                 "How many black dots are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-7dots", r, "7"),
                 "Counting"),
        TestCase("counting-squares", lambda: grid_img,
                 "How many coloured squares are in this image? Reply with just the number.",
                 5, lambda r: assert_number_in("counting-squares", r, "6"),
                 "Counting"),
        # Colour
        TestCase("colour-naming", lambda: red_img,
                 "What colour is this image? Reply with just the colour name.",
                 8, lambda r: assert_contains("colour-naming", r, "red"),
                 "Colour"),
        TestCase("colour-list", lambda: grid_img,
                 "List the distinct colours of all the squares in this image.",
                 60, lambda r: assert_colour_count("colour-list", r,
                     ["red","green","blue","orange","purple","yellow"], min_match=4),
                 "Colour"),
        # Spatial
        TestCase("spatial-above-below", lambda: spatial_img,
                 "Is the red square above or below the blue square? Reply: 'above' or 'below'.",
                 10, lambda r: assert_contains("spatial-above-below", r, "above"),
                 "Spatial"),
        TestCase("spatial-left-right", lambda: spatial_img,
                 "Is the blue square to the left or right of the red square? Reply: 'left' or 'right'.",
                 10, lambda r: assert_contains("spatial-left-right", r, "right"),
                 "Spatial"),
        TestCase("spatial-three-shapes", lambda: three_img,
                 "There are three coloured squares in this image. Which colour is in the centre? "
                 "Reply with just the colour name.",
                 10, lambda r: assert_contains("spatial-three-shapes", r, "green"),
                 "Spatial"),
        # Scene
        TestCase("scene-type", lambda: chart_img,
                 "Is this a chart/graph or a real-world photograph? Reply: 'chart' or 'photo'.",
                 5, lambda r: assert_contains("scene-type", r, "chart"),
                 "Scene"),
    ]

    # Real-world fixture: Wikimedia bar chart
    real_chart = _try_fixture("wikimedia-bar-chart")
    if real_chart is not None:
        _rc = real_chart
        tests.append(TestCase(
            "real-chart-type", lambda: _rc,
            "Is this image a bar chart or a real-world photograph? Reply: 'chart' or 'photo'.",
            5, lambda r: assert_contains("real-chart-type", r, "chart"),
            "Chart",
        ))

    # Real-world OCR: stop sign in photographic context
    real_stop = _try_fixture("wikimedia-stop-sign")
    if real_stop is not None:
        _rs = real_stop
        tests.append(TestCase(
            "real-ocr-sign", lambda: _rs,
            "What word is written on the sign in this image? Reply with just the word.",
            10, lambda r: assert_contains("real-ocr-sign", r, "STOP"),
            "OCR",
        ))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

DIMENSION_ORDER = ["OCR", "Chart", "Counting", "Colour", "Spatial", "Scene"]


def run_suite(label: str, tests: list[TestCase], base_url: str,
              timeout: int, pass_threshold: float) -> tuple[bool, dict]:
    """
    Run all tests, collect per-dimension accuracy + latency.
    Returns (passed_threshold, results_dict).
    """
    dims: dict[str, DimResult] = {d: DimResult() for d in DIMENSION_ORDER}
    all_passed = 0
    all_total = 0

    for tc in tests:
        dim = tc.dimension
        if dim not in dims:
            dims[dim] = DimResult()
        dims[dim].total += 1
        all_total += 1
        print(f"    {tc.dimension:<12} {tc.label:<30} ... ", end="", flush=True)
        try:
            img = tc.image_fn()
            response, elapsed_ms = ask(base_url, img, tc.prompt, tc.max_tokens, timeout)
            tc.assertion(response)
            dims[dim].passed += 1
            dims[dim].latencies_ms.append(elapsed_ms)
            all_passed += 1
            print(f"ok  ({elapsed_ms:.0f}ms)")
        except AssertionError as e:
            note = str(e)[:90]
            print(f"WARN  {note}")
            dims[dim].latencies_ms.append(0)
        except Exception as e:
            print(f"ERROR  {e}")
            dims[dim].latencies_ms.append(0)

    # ── Per-dimension summary ────────────────────────────────────────────────
    print()
    print(f"    {'Dimension':<16} {'Accuracy':<12} {'Avg latency'}")
    print(f"    {'-'*16} {'-'*12} {'-'*12}")
    for dim_name in DIMENSION_ORDER:
        dr = dims.get(dim_name)
        if dr is None or dr.total == 0:
            continue
        pct    = int(100 * dr.passed / dr.total)
        status = "✅" if dr.passed == dr.total else ("⚠️ " if dr.passed > 0 else "❌")
        avg_ms = dr.avg_latency()
        print(f"    {status} {dim_name:<14} {dr.passed}/{dr.total} ({pct:3d}%)   {avg_ms:6.0f} ms/q")

    # ── Overall score ────────────────────────────────────────────────────────
    score  = all_passed / all_total if all_total else 1.0
    pct    = int(score * 100)
    passed = score >= pass_threshold
    status = "✅" if passed else "❌"
    all_lats = [l for d in dims.values() for l in d.latencies_ms]
    overall_avg = sum(all_lats) / len(all_lats) if all_lats else 0
    print(f"    {'─'*48}")
    print(f"    {status} {label}: {all_passed}/{all_total} ({pct}%)   "
          f"avg {overall_avg:.0f} ms/q  [threshold: {int(pass_threshold*100)}%]")

    results = {
        "total": all_total,
        "passed": all_passed,
        "score_pct": pct,
        "avg_latency_ms": round(overall_avg, 1),
        "dimensions": {
            d: {
                "passed": dims[d].passed,
                "total": dims[d].total,
                "avg_ms": round(dims[d].avg_latency(), 1),
            }
            for d in DIMENSION_ORDER
            if d in dims and dims[d].total > 0
        },
    }
    return passed, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("base_url")
    parser.add_argument("--timeout",        type=int,   default=90)
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    args = parser.parse_args()

    if not HAVE_NUMPY:
        print("  [vision] numpy not available — skipping capability benchmarks")
        sys.exit(0)

    name = args.model_name
    if   "FastVLM" in name:
        tests = build_fastvlm_suite()
    elif "olmOCR" in name:
        tests = build_olmocr_suite()
    elif "Qwen3.6" in name:
        tests = build_qwen36_suite()
    elif "Qwen3" in name and "VL" in name:
        tests = build_qwen3vl_suite()
    elif "Qwen"  in name and "VL" in name:
        tests = build_qwen25vl_suite()
    else:
        print(f"  [vision] no capability suite for {name!r} — skipping")
        sys.exit(0)

    ok, _ = run_suite(name, tests, args.base_url, args.timeout, args.pass_threshold)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
