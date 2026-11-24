#!/usr/bin/env python3
"""Decision-LLM: write an answer word by word, with Jev making every decision.

How one word is produced (all decisions are Jev's):
  1. Letter step: one call, two questions. "What comes next?" (a word, punctuation,
     or stop) and "What is the first letter of the next word?" (A-Z only).
  2. Word tournament: every dictionary word starting with that letter (and the
     runner-up letter if Jev was unsure) is split into groups of up to 254. Jev picks
     its top 2 from each group in parallel, then picks the final word from the winners.

Why not spell every letter? Tests showed Jev is excellent at choosing whole words
and good at first letters, but weak at spelling letters 2-3 of a word.

Usage:
  python decision_llm.py --check                      # verify the API key
  python decision_llm.py "What is the capital of France?"
  python decision_llm.py --test                       # next-word accuracy test
  python decision_llm.py --mock --test                # run the plumbing without the API

Key: set TYPESAFE_API_KEY in your shell or in a .env file next to this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
API_BASE = os.environ.get("JEV_API_BASE", "https://api.typesafe.ai")
MODEL = os.environ.get("JEV_MODEL", "jev-latest")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = 254          # Jev allows 255 options per choice question
KEEP_PER_GROUP = 2         # tournament: how many words survive from each group
LETTER_COVERAGE = 0.75     # try first letters until their probabilities add up to this
MAX_LETTERS = 4

PUNCTUATION = {
    "period": (".", "End the sentence with a period"),
    "comma": (",", "Put a comma here"),
    "question_mark": ("?", "End the sentence with a question mark"),
}


# --------------------------------------------------------------------------- dictionary
class Dictionary:
    """A spelling list: which known words start with a letter. Never predicts."""

    def __init__(self, path: Path, vocab: int):
        if not path.exists():
            sys.exit(f"{path} not found. Run: .venv/bin/python build_dictionary.py")
        self.words = [w.strip() for w in path.read_text().splitlines() if w.strip()][:vocab]
        self.by_letter: dict[str, list[str]] = {}
        for w in self.words:  # stays in frequency order
            self.by_letter.setdefault(w[0].upper(), []).append(w)

    def __contains__(self, word: str) -> bool:
        return word in self.by_letter.get(word[0].upper(), [])

    def starting_with(self, letter: str) -> list[str]:
        return self.by_letter.get(letter, [])


# --------------------------------------------------------------------------- Jev client
Ranked = list[tuple[str, float]]  # (option, probability), best first


@dataclass
class Stats:
    calls: int = 0
    input_tokens: int = 0
    seconds: float = 0.0


class Jev:
    """Calls TypeSafe's Jev directly: POST /v1/systemone."""

    def __init__(self, api_key: str, model: str = MODEL):
        self.api_key = api_key
        self.model = model
        self.stats = Stats()
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            API_BASE + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    wait = float(e.headers.get("retry-after") or 2**attempt)
                    print(f"  [jev {e.code}, retrying in {wait:.0f}s]", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise SystemExit(f"Jev API error {e.code}: {detail}")
            except urllib.error.URLError as e:
                if attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise SystemExit(f"Could not reach Jev API: {e.reason}")
        raise RuntimeError("unreachable")

    def models(self) -> dict:
        return self._request("GET", "/v1/models")

    def ask(self, state: str, questions: dict[str, tuple[str, dict[str, str]]],
            ctx: "Context") -> dict[str, Ranked]:
        """Ask several choice questions about one state in one call.
        questions: {id: (instructions, {option: description})}."""
        body = {
            "model": self.model,
            "state": state,
            "questions": {qid: {"type": "choice", "instructions": instr, "criteria": crit}
                          for qid, (instr, crit) in questions.items()},
        }
        start = time.time()
        data = self._request("POST", "/v1/systemone", body)
        with self._lock:
            self.stats.calls += 1
            ctx.calls += 1
            self.stats.seconds += time.time() - start
            self.stats.input_tokens += (data.get("usage") or {}).get("input_tokens", 0)
        out = {}
        for qid in questions:
            a = data["answers"][qid]
            probs = a.get("probabilities") or {a["choice"]: 1.0}
            out[qid] = sorted(probs.items(), key=lambda kv: -(kv[1] or 0))
        return out


class MockJev(Jev):
    """Offline stand-in that 'knows' the intended word. Tests the loop, not Jev."""

    def __init__(self):
        self.stats = Stats()
        self._lock = threading.Lock()

    def ask(self, state, questions, ctx):
        with self._lock:
            self.stats.calls += 1
            ctx.calls += 1
        target = ctx.target or ""
        out = {}
        for qid, (_, crit) in questions.items():
            prefs = [target, target[:1].upper(), "word", "end"]
            pick = next((p for p in prefs if p in crit), next(iter(crit)))
            out[qid] = [(pick, 0.9)] + [(k, 0.1 / len(crit)) for k in crit if k != pick]
        return out


# --------------------------------------------------------------------------- generation
@dataclass
class Context:
    question: str
    tokens: list[str] = field(default_factory=list)  # words and punctuation so far
    target: str | None = None                         # only used by MockJev
    calls: int = 0                                    # Jev calls made for this context

    def answer_text(self) -> str:
        out = ""
        for t in self.tokens:
            out += t if t in ".,?!" or not out else " " + t
        return out

    def state(self) -> str:
        so_far = self.answer_text()
        blank = f"{so_far} ___" if so_far else "___"
        return (f"Question: {self.question}\n"
                f'Answer being written: "{blank}"\n'
                "The answer must be correct, grammatical English that answers the question. "
                "The blank (___) is the single next word or punctuation mark.")


@dataclass
class WordResult:
    token: str          # the word, punctuation, or "<end>"
    letters: Ranked     # Jev's first-letter ranking
    calls: int
    seconds: float
    next: Ranked = field(default_factory=list)    # word / punctuation / end ranking
    tried: list[str] = field(default_factory=list)  # first letters entered in the tournament
    candidates: int = 0                           # words in the tournament
    final: Ranked = field(default_factory=list)   # final-round ranking


def first_step(jev: Jev, ctx: Context) -> tuple[Ranked, Ranked]:
    """One call: what kind of thing comes next, and the next word's first letter."""
    kinds = {"word": "A word goes in the blank"}
    last = ctx.tokens[-1] if ctx.tokens else None
    if last and last not in ".,?!":
        kinds.update({k: desc for k, (_, desc) in PUNCTUATION.items()})
    if ctx.tokens:
        kinds["end"] = "Nothing goes in the blank: the answer is already a complete sentence"
    questions = {
        "letter": ("What is the first letter of the word that goes in the blank?",
                   {c: f"The letter {c}" for c in LETTERS}),
    }
    if len(kinds) > 1:
        questions["next"] = ("What goes in the blank?", kinds)
    res = jev.ask(ctx.state(), questions, ctx)
    return res.get("next", [("word", 1.0)]), res["letter"]


def pick_word(jev: Jev, ctx: Context, candidates: list[str], pool: ThreadPoolExecutor,
              verbose: bool) -> Ranked:
    """Tournament over candidates: groups of MAX_OPTIONS in parallel until one group is left."""
    instr = "Which word goes in the blank so the answer reads as correct, grammatical English?"
    rnd = 1
    while True:
        groups = [candidates[i:i + MAX_OPTIONS] for i in range(0, len(candidates), MAX_OPTIONS)]
        results = list(pool.map(
            lambda g: jev.ask(ctx.state(), {"w": (instr, {w: w for w in g})}, ctx)["w"], groups))
        if len(groups) == 1:
            if verbose:
                print(f"    [final  {len(candidates):4} words] {fmt(results[0])}", file=sys.stderr)
            return results[0]
        candidates = [w for r in results for w, _ in r[:KEEP_PER_GROUP]]
        if verbose:
            print(f"    [round {rnd} {len(groups):2} groups] -> {len(candidates)} finalists: "
                  f"{', '.join(candidates[:12])}{'...' if len(candidates) > 12 else ''}",
                  file=sys.stderr)
        rnd += 1


def fmt(r: Ranked, n: int = 3) -> str:
    return "  ".join(f"{k}={v:.2f}" for k, v in r[:n])


def generate_word(jev: Jev, d: Dictionary, ctx: Context, pool: ThreadPoolExecutor,
                  verbose: bool = False) -> WordResult:
    calls0, t0 = ctx.calls, time.time()
    nxt, letters = first_step(jev, ctx)
    if verbose:
        print(f"    [next] {fmt(nxt)}   [letter] {fmt(letters)}", file=sys.stderr)
    kind = nxt[0][0]
    chosen, candidates, final = [], [], []
    if kind == "end":
        token = "<end>"
    elif kind in PUNCTUATION:
        token = PUNCTUATION[kind][0]
    else:
        covered = 0.0
        for c, p in letters[:MAX_LETTERS]:
            chosen.append(c)
            covered += p
            if covered >= LETTER_COVERAGE:
                break
        if verbose:
            print(f"    [letters tried] {', '.join(chosen)}", file=sys.stderr)
        last = ctx.tokens[-1] if ctx.tokens else None
        candidates = [w for c in chosen for w in d.starting_with(c) if w != last]
        final = pick_word(jev, ctx, candidates, pool, verbose)
        token = final[0][0]
    if token != "<end>":
        ctx.tokens.append(token)
    return WordResult(token, letters, ctx.calls - calls0, time.time() - t0,
                      nxt, chosen, len(candidates), final)


def answer(jev: Jev, d: Dictionary, question: str, max_words: int, verbose: bool):
    ctx = Context(question)
    t0 = time.time()
    print(f"Q: {question}\nA: ", end="", flush=True)
    with ThreadPoolExecutor(16) as pool:
        for _ in range(max_words):
            r = generate_word(jev, d, ctx, pool, verbose)
            if r.token == "<end>":
                break
            if verbose:
                print(f"  -> {ctx.answer_text()}   ({r.calls} calls, {r.seconds:.1f}s)",
                      file=sys.stderr)
            else:
                sep = "" if r.token in ".,?!" or len(ctx.tokens) == 1 else " "
                print(sep + r.token, end="", flush=True)
    print()
    s = jev.stats
    print(f"\n[{s.calls} Jev calls, {time.time() - t0:.1f}s total, {s.input_tokens} input tokens]",
          file=sys.stderr)


# --------------------------------------------------------------------------- test
TEST_CASES = [
    ("What is the capital of France?", "The capital of France is", "paris"),
    ("What color is the sky on a clear day?", "The sky is", "blue"),
    ("How many legs does a spider have?", "A spider has", "eight"),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William", "shakespeare"),
    ("What is the largest planet in our solar system?", "The largest planet is", "jupiter"),
    ("What do bees make?", "Bees make", "honey"),
    ("What is frozen water called?", "Frozen water is called", "ice"),
    ("What gas do plants absorb from the air?", "Plants absorb carbon", "dioxide"),
    ("What is the opposite of hot?", "The opposite of hot is", "cold"),
    ("Which animal is called the king of the jungle?", "The king of the jungle is the", "lion"),
    ("What language is spoken in Brazil?", "People in Brazil speak", "portuguese"),
    ("How many days are in a week?", "A week has", "seven"),
    ("Which planet is known as the Red Planet?", "The Red Planet is", "mars"),
    ("What is the tallest mountain in the world?", "The tallest mountain in the world is Mount", "everest"),
    ("What do caterpillars turn into?", "Caterpillars turn into", "butterflies"),
    ("What is the largest ocean on Earth?", "The largest ocean on Earth is the", "pacific"),
]


def run_test(jev: Jev, d: Dictionary, verbose: bool):
    cases = [c for c in TEST_CASES if c[2] in d]
    for *_, e in TEST_CASES:
        if e not in d:
            print(f"(skipping {e}: not in dictionary)")

    def one(case):
        q, so_far, expected = case
        ctx = Context(q, tokens=so_far.lower().split(), target=expected)
        with ThreadPoolExecutor(16) as pool:
            return expected, generate_word(jev, d, ctx, pool, verbose)

    t0 = time.time()
    with ThreadPoolExecutor(4) as outer:  # a few test cases at a time
        results = list(outer.map(one, cases))

    print(f"{'expected':<12} {'got':<12} {'1st letter':<14} calls  secs")
    letter_ok = word_ok = calls = 0
    for expected, r in results:
        top = r.letters[0][0]
        l_ok, w_ok = top == expected[0].upper(), r.token == expected
        letter_ok += l_ok
        word_ok += w_ok
        calls += r.calls
        mark = "OK" if w_ok else "X "
        print(f"{expected:<12} {r.token:<12} {('OK ' if l_ok else 'X  ') + f'{top}={r.letters[0][1]:.2f}':<14}"
              f" {r.calls:>5}  {r.seconds:4.1f}  {mark}")
    n = len(results)
    print(f"\nfirst letter correct: {letter_ok}/{n}   word correct: {word_ok}/{n}   "
          f"avg calls/word: {calls / max(n, 1):.1f}   total time: {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- main
def load_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    env_file = HERE / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Set TYPESAFE_API_KEY in your shell or in .env (TYPESAFE_API_KEY=...)")
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--test", action="store_true", help="run the next-word accuracy test")
    ap.add_argument("--check", action="store_true", help="verify the API key via GET /v1/models")
    ap.add_argument("--mock", action="store_true", help="use an offline fake Jev (plumbing test)")
    ap.add_argument("--vocab", type=int, default=20000, help="dictionary size (most common words)")
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("-v", "--verbose", action="store_true", help="show every Jev decision")
    args = ap.parse_args()

    d = Dictionary(HERE / "words.txt", args.vocab)
    jev = MockJev() if args.mock else Jev(load_key())

    if args.check:
        print(json.dumps(jev.models(), indent=2))
    elif args.test:
        run_test(jev, d, args.verbose)
    elif args.question:
        answer(jev, d, args.question, args.max_words, args.verbose)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
