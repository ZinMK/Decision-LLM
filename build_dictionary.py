#!/usr/bin/env python3
"""Build words.txt: the most common English words, most frequent first.

The dictionary is only a spelling list. It never predicts anything. Jev makes
every decision; the dictionary just tells us which words a prefix can still
become. Run once:  .venv/bin/python build_dictionary.py
"""
import re
from pathlib import Path

from wordfreq import top_n_list

N = 50_000
OUT = Path(__file__).parent / "words.txt"

words = [w for w in top_n_list("en", N) if re.fullmatch(r"[a-z]+", w)]
OUT.write_text("\n".join(words) + "\n")
print(f"wrote {len(words)} words to {OUT}")
