#!/usr/bin/env python3
"""Decision-LLM: Jev writes an answer one letter at a time.

Each word: Jev types the first 2 letters itself, one letter question each. Then Jev is shown
every dictionary word that starts with those letters (in groups of 254, best ones go to a
final round) and picks the word it means; the rest of that word is added in one step. If no
word fits, Jev can say so and the last letter is deleted. At the start of a word, Jev can
instead say the sentence is finished (accepted only if Jev also says the reply is complete).

Each letter step is one Jev call about the answer so far, with up to three questions:
  - "What is the next letter?"        a-z, or backspace to undo the last character
    (the letter question also has a "word is done" option: picked -> we add a space ...)
  - "Is the sentence so far a sensible start of a correct answer?"
                                      ... unless this says no: then the whole word is
                                      deleted and blocked in that spot (after 4 rejections
                                      in one spot, the word before is removed too)
  - "Is the answer complete?"         yes -> we add a period and stop

Helpers (Jev still makes every choice):
  - Spelling guard: only letters that can still lead to a real English word are offered,
    and the yes/no questions are only asked once the current word is a real word.
  - Backspace: only used when Jev is at least 50% sure, and a character Jev deleted is
    not offered again in the same spot.
  - Progress is locked: an accepted word (complete + makes sense) can never be deleted,
    and a letter Jev picked with 80%+ confidence can't be backspaced (only the
    whole-word check can remove it).

The same loop can run on Cloudflare's Clef-omni instead of Jev (same question format).

Usage:
  python decision_llm.py "What is the capital of France?"
  python decision_llm.py -v "What is the capital of France?"   # show each decision
  python decision_llm.py --model clef "What is the capital of France?"

Keys, in your shell or in a .env file next to this script:
  Jev:   TYPESAFE_API_KEY
  Clef:  CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (a token with Workers AI access)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).parent
API_URL = os.environ.get("JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
MODEL = os.environ.get("JEV_MODEL", "jev-latest")
CLEF_MODEL = os.environ.get("CLEF_MODEL", "clef-omni")
CLEF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"

LETTERS = "abcdefghijklmnopqrstuvwxyz"
BACKSPACE = "backspace"
SPACE = "space"   # step key when Jev says the word is complete
DELETE_WORD = "delete_word"  # step key when a finished word doesn't make sense
DONE = "done"     # step key when Jev says the answer is complete
YES = 0.5         # yes/no threshold
BACKSPACE_MIN = 0.5  # only delete when Jev is this sure; weak backspaces are second-guessing
LOCK_AT = 0.8        # a letter Jev picks this confidently is locked: backspace can't remove it
FINISHED = "finished"  # letter option: "nothing to add, this word is complete"
STUCK_AFTER = 4        # after this many rejected words in one spot, also remove the word before
PREVIEW_CHARS = 30
EXAMPLES = 5        # example words shown with each letter option (spelling only, by frequency)
EXAMPLE_POOL = 12
PICK = "pick"          # step key when Jev picks the planned word
NONE_FIT = "none_fit"  # word-pick option: no word with this start fits
GROUP_SIZE = 254       # Jev allows 255 options per question
GROUPS_PER_CALL = 4    # word-pick groups per call, the same for every model (speed only)
FINALIST_P = 0.2       # besides each group's winner, runners-up this likely also go to the final
PREFIX_LETTERS = 2     # Jev types this many letters of each word before it picks the word
COMMON_POOL = 2000     # ...plus the 2000 most common words, whatever letter they start with,
                       #   plus the words of the user's question
END = "end"            # letter option at a word start: the sentence is finished
MAX_TOKENS = 5_000_000 # budget per answer: stop once Jev has read this many input tokens
# The word list has fragments like "c", "th", "pa". Words of 1-2 letters must be in this list.
SHORT_WORDS = set("""a i am an as at be by do go he hi if in is it me mr ms my no of oh ok on or
                    so to tv uk up us we dr pm ox""".split())

# Works for questions ("What do bees make?") and requests ("Give me a sentence to brighten my day").
GOAL = "a good reply to the user's message (accurate if it asks a question)"

PICK_GROUP_INSTRUCTIONS = ('Which word should come next, right after "{before}"? Pick the word '
                           "that keeps the sentence grammatical and makes it " + GOAL + ".")
FIRST_WORD_NOTE = (" This is the first word. A full-sentence reply starts like a sentence, "
                   "usually with its subject (for example \"The ...\", \"You ...\" or a name), "
                   "not with a bare one-word answer.")
PICK_FINAL_INSTRUCTIONS = ("Which whole word should come next? Each option is a complete word "
                           "exactly as it will be written, not the start of a longer word. Pick "
                           "the word that keeps the sentence grammatical and makes it "
                           + GOAL + ". If the sentence is already finished, pick the "
                           "option that ends it. If none of them fits, pick that option.")

LETTER_INSTRUCTIONS = ("What should the next letter be? Pick the option that keeps the sentence "
                       "grammatical and correctly spelled, so it can grow into a complete sentence "
                       "that is " + GOAL + ". If the last letter is a mistake, pick "
                       "backspace. If the current word is already complete, pick that option.")
FIRST_LETTER_INSTRUCTIONS = ("What is the first letter of a complete, grammatical sentence that "
                             "is " + GOAL + "?")


# --------------------------------------------------------------------------- spelling guard
class Spelling:
    """Knows which letter sequences can still become a real word. Never predicts."""

    def __init__(self, path: Path = HERE / "words.txt"):
        if not path.exists():
            sys.exit(f"{path} not found. Run: .venv/bin/python build_dictionary.py")
        ordered = [w.strip() for w in path.read_text().splitlines() if w.strip()]  # most common first
        ordered = [w for w in ordered if len(w) > 2 or w in SHORT_WORDS]
        self.common = ordered[:COMMON_POOL]
        self.words = set(ordered)
        self.prefix_count: dict[str, int] = defaultdict(int)  # prefix -> words starting with it
        self.examples: dict[str, list[str]] = defaultdict(list)  # prefix -> most common words
        self.by_first: dict[str, list[str]] = defaultdict(list)  # first letter -> words, by frequency
        for w in ordered:
            self.by_first[w[0]].append(w)
            for i in range(1, len(w) + 1):
                self.prefix_count[w[:i]] += 1
                if len(self.examples[w[:i]]) < EXAMPLE_POOL:
                    self.examples[w[:i]].append(w)

    def candidates(self, prefix: str, blocked_words: set[str]) -> list[str]:
        """Every word that starts with prefix, most common first."""
        return [w for w in self.by_first.get(prefix[:1], [])
                if w.startswith(prefix) and w not in blocked_words]

    def examples_for(self, prefix: str, blocked_words: set[str]) -> list[str]:
        return [w for w in self.examples.get(prefix, []) if w not in blocked_words][:EXAMPLES]

    @staticmethod
    def current_word(answer: str) -> str:
        return answer.split(" ")[-1]

    def can_add(self, answer: str, letter: str, blocked_words: set[str] = frozenset()) -> bool:
        """True if some real word, other than a blocked one, starts with word + letter."""
        p = self.current_word(answer) + letter
        return self.prefix_count.get(p, 0) > sum(1 for w in blocked_words if w.startswith(p))

    def is_word(self, answer: str, blocked_words: set[str] = frozenset()) -> bool:
        word = self.current_word(answer)
        return word in self.words and word not in blocked_words


# --------------------------------------------------------------------------- Jev
class Jev:
    """TypeSafe's Jev, direct API. Clef (below) has the same interface."""
    name = "Jev"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls = 0
        self.input_tokens = 0
        self._lock = threading.Lock()  # word-pick groups are asked in parallel

    def request(self, state: str, questions: dict[str, dict]) -> urllib.request.Request:
        body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()
        return urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    def ask(self, state: str, questions: dict[str, dict]) -> dict[str, dict]:
        """Ask several questions about one state in one call. Returns the answers by id."""
        req = self.request(state, questions)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4:
                    time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
                    continue
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"{self.name} API error {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Could not reach {self.name} API: {getattr(e, 'reason', e)}")
        data = data.get("result", data)  # Cloudflare wraps the answer in "result"
        with self._lock:
            self.calls += 1
            self.input_tokens += (data.get("usage") or {}).get("input_tokens", 0)
        return data["answers"]


class Clef(Jev):
    """Cloudflare's Clef-omni on Workers AI. Same questions and answers as Jev."""
    name = "Clef-omni"

    def __init__(self, account_id: str, api_token: str):
        super().__init__(api_token)
        self.url = CLEF_URL.format(account=account_id, model=CLEF_MODEL)

    def request(self, state: str, questions: dict[str, dict]) -> urllib.request.Request:
        body = json.dumps({"model": CLEF_MODEL, "state": state, "questions": questions}).encode()
        return urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})


def ranked(answer: dict) -> list[tuple[str, float]]:
    probs = answer.get("probabilities") or {answer["choice"]: 1.0}
    return sorted(probs.items(), key=lambda kv: -(kv[1] or 0))


# --------------------------------------------------------------------------- the loop
def display(answer: str) -> str:
    """Capitalize the first letter (Jev picks lowercase letters)."""
    return answer[:1].upper() + answer[1:]


def build_state(question: str, answer: str, target: str | None = None) -> str:
    """What Jev sees: the task, the finished words, and the word being spelled."""
    *done, current = answer.split(" ")
    done_words = [w for w in done if w]
    lines = [
        f"User's message: {question}",
        "Task: reply in one complete, grammatical English sentence. If the message asks a "
        "question, answer it accurately. If it asks for something, such as a kind or funny "
        "sentence, write that sentence. The sentence is written one letter at a time.",
        f'Sentence so far: "{display(answer).strip()}"' if answer else "Sentence so far: (nothing yet)",
    ]
    if done_words:
        numbered = "  ".join(f"{i}. {w}" for i, w in enumerate(done_words, 1))
        lines.append(f"Finished words: {numbered}")
    n = len(done_words) + 1
    if current and target and target.startswith(current):
        lines.append(f'Word {n} is being spelled: "{current}" (planned word: "{target}")')
    elif current:
        lines.append(f'Word {n} is being spelled: "{current}" (it may still be unfinished)')
    else:
        lines.append(f"Next: the first letter of word {n}"
                     + (f', which comes right after "{done_words[-1]}"' if done_words else ""))
    return "\n".join(lines)


def tail(text: str) -> str:
    return ("…" if len(text) > PREVIEW_CHARS else "") + text[-PREVIEW_CHARS:]


def letter_options(answer: str, spelling: Spelling, blocked: set[str],
                   blocked_words: set[str], locked: int, word_can_end: bool,
                   target: str | None = None, allow_end: bool = False) -> dict[str, str]:
    """Each option shows what the answer would look like after it."""
    word = spelling.current_word(answer)
    on_plan = bool(target) and target.startswith(word)
    nxt = target[len(word)] if on_plan and len(target) > len(word) else None
    opts = {}
    for c in LETTERS:
        if c in blocked or not spelling.can_add(answer, c, blocked_words):
            continue
        if c == nxt:
            hint = f'continues the planned word "{target}"'
        else:
            hint = "spelling a word like: " + ", ".join(spelling.examples_for(word + c, blocked_words))
        opts[c] = f'Add the letter "{c}" → "{tail(display(answer + c))}" ({hint})'

    if allow_end and not word and answer.strip():
        opts[END] = f'End the sentence here, it is finished → "{tail(display(answer.rstrip()))}."'
    if word_can_end:  # so Jev never has to use backspace to say "this word is done"
        what = "the complete planned word" if word == target else "already a complete word"
        opts[FINISHED] = f'Nothing to add: "{word}" is {what} → "{tail(display(answer))} "'
    # Locked progress can't be backspaced, unless only one option is left: every choice needs
    # at least two options (Clef-omni rejects a choice with one).
    if len(answer) > locked or (answer and len(opts) == 1):
        opts[BACKSPACE] = (f'Delete the last character because it is a mistake → '
                           f'"{tail(display(answer[:-1]))}"')
    if word and len(opts) == 1:  # only "delete a letter" is left: also offer the whole word
        opts[DELETE_WORD] = (f'Delete the whole word "{word}" because it is a mistake → '
                             f'"{tail(display(answer[:len(answer) - len(word)]))}"')
    return opts


def yes_no(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def complete_question(sentence: str) -> dict:
    """"Is the reply complete?" Asked after each finished word, and before "end the sentence"
    is accepted."""
    return yes_no(f'Is "{sentence}" a complete, grammatical sentence that is {GOAL}, '
                  "with nothing missing, so the reply is finished?")


def pick_word(jev: Jev, question: str, answer: str, spelling: Spelling,
              blocked_words: set[str], allow_none: bool,
              starts: list[str] | None = None,
              blocked_next: set[str] = frozenset(),
              allow_end: bool = True) -> tuple[list[tuple[str, float]], int]:
    """Jev picks the word it is spelling from every dictionary word that starts the same way
    (or, at the start of a word, with any of the letters in `starts`).

    Up to 254 words: one question. More: groups of 254 are asked in parallel, and each
    group's best words go to a final round. Returns the final round (best first) and how
    many words Jev chose from.
    """
    word = spelling.current_word(answer)
    before = answer[:len(answer) - len(word)]
    state = build_state(question, answer)
    words = [w for s in (starts or [word]) for w in spelling.candidates(s, blocked_words)]
    if starts:  # a new word: common words and the question's words compete too
        asked = [w for w in re.findall(r"[a-z]+", question.lower()) if w in spelling.words]
        words = list(dict.fromkeys(asked + words + spelling.common))
        words = [w for w in words if w not in blocked_words]
    # Skip words whose next letter was deleted in this spot, or that would have to end here
    # when ending here was undone (otherwise the same pick and delete repeat forever).
    n = len(word)
    words = [w for w in words if (w[n] not in blocked_next if len(w) > n else SPACE not in blocked_next)]
    pool = len(words)
    if len(words) > GROUP_SIZE - 1:
        instructions = PICK_GROUP_INSTRUCTIONS.format(before=display(before).strip())
        size = -(-len(words) // -(-len(words) // GROUP_SIZE))  # even groups, none of 1 word
        groups = [(f"g{i}", {"type": "choice", "instructions": instructions,
                             "criteria": {w: w for w in words[j:j + size]}})
                  for i, j in enumerate(range(0, len(words), size))]
        calls = [dict(groups[i:i + GROUPS_PER_CALL])
                 for i in range(0, len(groups), GROUPS_PER_CALL)]
        with ThreadPoolExecutor(len(calls)) as executor:
            results = list(executor.map(lambda qs: jev.ask(state, qs), calls))
        finalists = []
        for res in results:
            for _, answer_ in sorted(res.items(), key=lambda kv: int(kv[0][1:])):
                best = ranked(answer_)
                finalists += [best[0][0]] + [w for w, p in best[1:3] if (p or 0) >= FINALIST_P]
        words = list(dict.fromkeys(finalists))[:GROUP_SIZE - 1]
    if not words:
        return [], 0
    criteria = {w: f'The whole word "{w}", exactly as written → "{tail(display(before + w))} …"'
                for w in words}
    if starts and before.strip() and allow_end:
        criteria[END] = f'End the sentence here, it is finished → "{tail(display(before.rstrip()))}."'
    if allow_none or len(criteria) < 2:  # every choice needs at least two options
        criteria[NONE_FIT] = (f'None of these words fits here, so no word should start with '
                              f'"{word}" → delete the "{word[-1]}"')
    instructions = PICK_FINAL_INSTRUCTIONS + (FIRST_WORD_NOTE if not before.strip() else "")
    res = jev.ask(state, {"word": {"type": "choice", "instructions": instructions,
                                   "criteria": criteria}})
    return ranked(res["word"]), pool


@dataclass
class Step:
    key: str                         # a letter, "backspace", "space" or "done"
    answer: str                      # answer after this step
    letters: list[tuple[str, float]] # Jev's top letter options
    word_complete: float | None      # P(yes) for "is the word complete?" (None = not asked)
    on_track: float | None           # P(yes) for "does the answer make sense so far?"
    answer_complete: float | None    # P(yes) for "is the answer complete?"
    seconds: float
    locked: int = 0                  # answer[:locked] is locked progress
    target: str | None = None        # the planned word being spelled
    pool: int = 0                    # word picks: how many words Jev chose from
    typed: str = ""                  # word picks: the letters Jev typed before picking


def generate(jev: Jev, question: str, max_chars: int = 300, spelling: Spelling | None = None,
             max_steps: int | None = None) -> Iterator[Step]:
    spelling = spelling or Spelling()
    max_steps = max_steps or max_chars * 3
    answer = ""
    locked = 0  # answer[:locked] can't be backspaced
    blocked: dict[str, set[str]] = defaultdict(set)  # answer -> characters Jev deleted there
    blocked_words: dict[str, set[str]] = defaultdict(set)  # answer before a word -> rejected words
    target = None  # the planned word Jev picked for the current word
    no_end: set[str] = set()  # answers where Jev said the reply is not complete yet
    budget = jev.input_tokens + MAX_TOKENS
    for _ in range(max_steps):
        if len(answer) >= max_chars or jev.input_tokens >= budget:
            return
        word = spelling.current_word(answer)
        word_start = answer[:len(answer) - len(word)]
        bad_words = blocked_words[word_start]
        if len(word) < PREFIX_LETTERS:
            target = None  # Jev is still typing the start of the word itself

        # Jev has typed the start of a word, and there is no picked word for it (or Jev typed
        # off it): let Jev pick the word from every word that starts this way, then add the rest.
        if len(word) >= PREFIX_LETTERS and not (target and target.startswith(word)
                                                and target not in bad_words):
            t0 = time.time()
            can_undo = len(answer) > locked
            words, pool = pick_word(jev, question, answer, spelling, bad_words, can_undo,
                                    blocked_next=blocked[answer])
            if words and words[0][0] != NONE_FIT:
                target = words[0][0]
                answer = word_start + target
                yield Step(PICK, answer, words[:6], None, None, None, time.time() - t0, locked, target,
                           pool, word)
                continue
            if not words and not can_undo:
                return  # nothing fits and the letter is locked: give up
            # No word fits: the last letter was a mistake.
            removed, answer = answer[-1], answer[:-1]
            blocked[answer].add(removed)
            locked = min(locked, len(answer))
            target = None
            yield Step(BACKSPACE, answer, words[:6], None, None, None, time.time() - t0, locked)
            continue

        word_can_end = bool(word) and spelling.is_word(answer, bad_words) and SPACE not in blocked[answer]
        criteria = letter_options(answer, spelling, blocked[answer], bad_words, locked, word_can_end,
                                  target, answer not in no_end)
        if len(criteria) < 2 and blocked[answer]:
            # Every choice needs at least two options: allow the deleted letters here again.
            blocked[answer].clear()
            criteria = letter_options(answer, spelling, blocked[answer], bad_words, locked,
                                      word_can_end, target, answer not in no_end)
        if not criteria:
            return  # Jev rejected every possible first letter: give up
        instructions = LETTER_INSTRUCTIONS if answer else FIRST_LETTER_INSTRUCTIONS
        if END in criteria:
            instructions += " If the sentence is already finished, pick the option that ends it."
        questions = {"letter": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        # Only ask the yes/no questions once the current word is a real word.
        if word_can_end:
            sentence = display(answer)
            questions["on_track"] = yes_no(
                f'Taking "{word}" as a complete word, is "{sentence}" a sensible, grammatical '
                "start of a sentence that is " + GOAL + "? It is fine that the "
                "sentence is not finished yet.")
            questions["answer_complete"] = complete_question(sentence)

        t0 = time.time()
        res = jev.ask(build_state(question, answer, target), questions)
        letters = ranked(res["letter"])
        # "Is the word complete?" is answered by the letter question itself: the "word is done"
        # option competes directly with adding another letter ("cap" vs "capi...").
        p_word = dict(letters).get(FINISHED) if word_can_end else None
        p_track = res["on_track"]["noul"] if "on_track" in res else None
        p_answer = res["answer_complete"]["noul"] if "answer_complete" in res else None

        word_done = word_can_end and letters[0][0] == FINISHED

        if word_can_end and p_answer >= YES and p_track >= YES and (word_done or word == target):
            key, answer = DONE, answer + "."
            locked = len(answer)
        elif word_done:
            if p_track >= YES:  # accepted word: lock it for good
                key, answer = SPACE, answer + " "
                locked = len(answer)
                target = None
            else:  # a finished word that doesn't make sense: delete it, block it here
                key, answer = DELETE_WORD, word_start
                bad_words.add(word)
                if len(bad_words) >= STUCK_AFTER and answer:
                    # Stuck: nothing fits after the previous word. Unlock it and remove it too,
                    # and don't allow it in its spot again.
                    prev = answer.rstrip(" ").split(" ")[-1]
                    answer = answer.rstrip(" ")[:-len(prev)]
                    blocked_words[answer].add(prev)
                locked = min(locked, len(answer))
        else:
            key, p = next((k, p) for k, p in letters if k != FINISHED)
            if key == END:
                # Jev picked "end the sentence". Accept it only if Jev also says the reply is
                # complete. Otherwise drop it and use Jev's next-best option.
                check = jev.ask(build_state(question, answer),
                                {"answer_complete": complete_question(display(answer.rstrip()))})
                p_answer = check["answer_complete"]["noul"]
                if p_answer >= YES:
                    answer = answer.rstrip() + "."
                    yield Step(DONE, answer, letters[:6], None, None, p_answer, time.time() - t0,
                               len(answer))
                    return
                no_end.add(answer)
                letters = [(k, q) for k, q in letters if k != END]
                key, p = next((k, p) for k, p in letters if k != FINISHED)
            if key == BACKSPACE and p < BACKSPACE_MIN:
                key, p = next(((k, p) for k, p in letters if k not in (BACKSPACE, FINISHED, DELETE_WORD)),
                              (key, p))
            if key == DELETE_WORD:  # nothing can follow: take the unfinished word back
                answer = word_start
                bad_words.add(target or word)
                blocked[answer].add(word[0])
                locked = min(locked, len(answer))
                target = None
            elif key == BACKSPACE:
                removed, answer = answer[-1], answer[:-1]
                blocked[answer].add(SPACE if removed == " " else removed)
                locked = min(locked, len(answer))
                target = None  # pick again from the words that still fit
            else:
                if p >= LOCK_AT and locked == len(answer):  # confident and contiguous: lock it
                    locked += 1
                answer += key
        yield Step(key, answer, letters[:6], p_word, p_track, p_answer, time.time() - t0, locked,
                   target)
        if key == DONE:
            return


# --------------------------------------------------------------------------- CLI
def env_value(name: str) -> str:
    """A setting from the shell, or from the .env file next to this script."""
    value = os.environ.get(name, "")
    env_file = HERE / ".env"
    if not value and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
    return value


def make_decider(model: str) -> Jev:
    """A Jev or Clef client. Raises RuntimeError if its key is missing."""
    if model == "clef":
        account, token = env_value("CLOUDFLARE_ACCOUNT_ID"), env_value("CLOUDFLARE_API_TOKEN")
        if not (account and token):
            raise RuntimeError("Clef-omni needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN "
                               "in .env (a token with Workers AI access).")
        return Clef(account, token)
    key = env_value("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("Jev needs TYPESAFE_API_KEY in .env.")
    return Jev(key)


def load_key() -> str:
    key = env_value("TYPESAFE_API_KEY")
    if not key:
        sys.exit("Set TYPESAFE_API_KEY in your shell or in .env (TYPESAFE_API_KEY=...)")
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("-v", "--verbose", action="store_true", help="show each decision")
    ap.add_argument("--model", choices=["jev", "clef"], default="jev", help="which model decides")
    args = ap.parse_args()

    try:
        jev = make_decider(args.model)
    except RuntimeError as e:
        sys.exit(str(e))
    t0 = time.time()
    print(f"Q: {args.question}")
    answer, backspaces = "", 0
    for step in generate(jev, args.question, args.max_chars):
        answer = step.answer
        backspaces += step.key == BACKSPACE
        if args.verbose:
            yn = lambda p: "  -  " if p is None else f"{p:.2f}"
            top = "  ".join(f"{k}={p:.2f}" for k, p in step.letters[:3])
            if step.key == PICK:
                print(f'  {"pick":>11}  planned word "{step.target}"   <- {top}', file=sys.stderr)
                continue
            print(f'  {step.key:>11}  word?{yn(step.word_complete)} sense?{yn(step.on_track)} '
                  f'done?{yn(step.answer_complete)}'
                  f'  "{display(answer)[:step.locked]}|{answer[step.locked:]}"   <- {top}',
                  file=sys.stderr)  # | marks the end of locked progress
        else:
            print(f"\r\033[K{display(answer)}", end="", file=sys.stderr, flush=True)
    print(f"\r\033[KA: {display(answer)}")
    print(f"[{jev.calls} {jev.name} calls, {backspaces} backspaces, {time.time() - t0:.1f}s, "
          f"{jev.input_tokens} input tokens]", file=sys.stderr)


if __name__ == "__main__":
    main()
