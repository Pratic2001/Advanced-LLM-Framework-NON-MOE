#!/usr/bin/env python3
"""
diagnose_formats.py

Check which content-format dependencies are actually usable before a real run.

Usage:
    python diagnose_formats.py
"""
import importlib
import shutil
import sys

CHECKS = [
    ("html", ["trafilatura", "readability"], []),
    ("pdf (text layer)", ["pdfplumber"], []),
    ("pdf (OCR fallback)", ["pdf2image", "pytesseract"], ["pdftoppm", "tesseract"]),
    ("docx", ["docx"], []),
    ("pptx", ["pptx"], []),
    ("xlsx / csv", ["openpyxl"], []),
    ("image OCR", ["PIL", "pytesseract"], ["tesseract"]),
    ("video captions", ["yt_dlp"], []),
    ("video/audio ASR fallback", ["yt_dlp", "faster_whisper"], ["ffmpeg"]),
]

print("Checking optional dependencies for each supported content format...\n")

any_missing = False
for label, modules, binaries in CHECKS:
    missing_modules = []
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            missing_modules.append(m)
    missing_binaries = [b for b in binaries if shutil.which(b) is None]

    if not missing_modules and not missing_binaries:
        print(f"  OK    {label}")
    else:
        any_missing = True
        parts = []
        if missing_modules:
            parts.append(f"pip install {' '.join(missing_modules)}")
        if missing_binaries:
            parts.append(f"install system packages: {', '.join(missing_binaries)}")
        print(f"  MISS  {label:<38} -- {'; '.join(parts)}")

print()
if any_missing:
    print("Formats marked MISS will return a clear error from extract_content "
          "instead of silently producing nothing. Install the missing deps "
          "for formats you actually need.")
else:
    print("All optional format dependencies are installed.")

sys.exit(1 if any_missing else 0)
