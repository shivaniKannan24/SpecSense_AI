# Extract structured constraints (budget, weight, use-case, soft prefs)
from pydantic import BaseModel
import re

class Constraints(BaseModel):
    hard: dict = {}
    soft: list = []


def extract_constraints(text: str) -> dict:
    """Lightweight extractor using regex + heuristics. Replace or extend with LLM calls."""
    text_l = text.lower()
    c = Constraints()

    # budget (e.g., 'under 30k', 'below 50000')
    m = re.search(r"under\s*(\d+[kKmM]?)", text_l)
    if m:
        c.hard['budget'] = m.group(1)

    if 'light' in text_l or 'lightweight' in text_l:
        c.hard['weight'] = 'light'

    # simple purpose mapping
    if 'coding' in text_l or 'programming' in text_l:
        c.hard['use_case'] = 'coding'
        c.soft.append('good_cpu')

    if 'battery' in text_l or 'battery backup' in text_l:
        c.soft.append('long_battery')

    return c.dict()

