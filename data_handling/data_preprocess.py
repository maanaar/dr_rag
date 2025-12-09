# data_handling/data_preprocess.py
import re
from .business_unit_map import map_business_unit

# Custom stopwords you requested
CUSTOM_STOPWORDS = {
    "انا", "عايز", "ابي", "ابغى", "محتاج", "لو", "من", "في", "عاوزه", "عاوز"
}

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.split(";")[0].strip()

def normalize_query(text):
    text = clean_text(text)
    words = [w for w in text.split() if w not in CUSTOM_STOPWORDS]
    return " ".join(words)

def expand_business_units(raw_bu):
    """
    Example input:
        "MKR;#10;#AFW"

    → extracts valid codes only:
        ["MKR", "AFW"]
    """
    if not raw_bu:
        return []

    parts = raw_bu.split(";")
    valid_codes = [p.replace("#", "").strip() for p in parts if p.replace("#", "").strip().isalpha()]

    return [map_business_unit(code) for code in valid_codes]
