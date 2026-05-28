"""
Smart Grammar Correction & Sentence Composer for Sign Language
===============================================================
Converts raw sign-word sequences into natural English sentences.
Uses spaCy for POS tagging + layered heuristic rules tuned for ISL/ASL output.

Sign language grammar typically:
  - Drops articles (a, an, the)
  - Omits copula (is, am, are)
  - Uses topic-comment structure  (STORE I GO → "I am going to the store")
  - Doesn't conjugate verbs
  - Uses different word order
  - Question words often come at end  (YOU NAME WHAT → "What is your name?")

This module applies:
  1. Exact phrase mapping (150+ common sequences)
  2. Pattern-based sentence templates (question, request, statement patterns)
  3. spaCy POS-tag based corrections
  4. Basic heuristic fallback
"""

import logging
import re
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ── Try loading spaCy ────────────────────────────────────────────────────
_nlp = None
_SPACY_OK = False
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_OK = True
    logger.info("Grammar correction: spaCy loaded")
except Exception:
    logger.warning("spaCy not available — using basic grammar correction")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  EXACT PHRASE MAP  — 150+ common sign sequences → natural English
# ═══════════════════════════════════════════════════════════════════════════

_PHRASE_MAP = {
    # ── Greetings & Farewells ─────────────────────────────────────────
    "hello":                    "Hello!",
    "hello how you":            "Hello, how are you?",
    "how you":                  "How are you?",
    "how you today":            "How are you today?",
    "good morning":             "Good morning!",
    "good night":               "Good night!",
    "good evening":             "Good evening!",
    "good afternoon":           "Good afternoon!",
    "goodbye":                  "Goodbye!",
    "see you":                  "See you!",
    "see you later":            "See you later!",
    "see you tomorrow":         "See you tomorrow!",
    "nice meet you":            "Nice to meet you!",

    # ── Thanks & Courtesy ────────────────────────────────────────────
    "thank you":                "Thank you.",
    "thank you help":           "Thank you for your help.",
    "thank you much":           "Thank you very much.",
    "please":                   "Please.",
    "please help":              "Please help me.",
    "please help me":           "Please help me.",
    "please wait":              "Please wait.",
    "please come":              "Please come here.",
    "please stop":              "Please stop.",
    "excuse me":                "Excuse me.",
    "sorry":                    "I'm sorry.",
    "sorry i late":             "I'm sorry I'm late.",
    "sorry i wrong":            "I'm sorry, I was wrong.",
    "you welcome":              "You're welcome.",
    "no problem":               "No problem.",

    # ── Self-descriptions (I + adjective) ────────────────────────────
    "i good":                   "I'm good.",
    "i fine":                   "I'm fine.",
    "i happy":                  "I'm happy.",
    "i sad":                    "I'm sad.",
    "i tired":                  "I'm tired.",
    "i sick":                   "I'm sick.",
    "i hungry":                 "I'm hungry.",
    "i thirsty":                "I'm thirsty.",
    "i angry":                  "I'm angry.",
    "i scared":                 "I'm scared.",
    "i busy":                   "I'm busy.",
    "i ready":                  "I'm ready.",
    "i late":                   "I'm late.",
    "i sorry":                  "I'm sorry.",
    "i cold":                   "I'm cold.",
    "i hot":                    "I'm hot.",
    "i ok":                     "I'm okay.",
    "i bad":                    "I'm not feeling well.",

    # ── "I + verb" patterns ──────────────────────────────────────────
    "i eat":                    "I'm eating.",
    "i drink":                  "I'm drinking.",
    "i go":                     "I'm going.",
    "i come":                   "I'm coming.",
    "i work":                   "I'm working.",
    "i think":                  "I'm thinking.",
    "i know":                   "I know.",
    "i understand":             "I understand.",
    "i love you":               "I love you.",
    "i miss you":               "I miss you.",
    "i like":                   "I like it.",
    "i like you":               "I like you.",
    "i need help":              "I need help.",
    "i need water":             "I need water.",
    "i need eat":               "I need to eat.",
    "i need go":                "I need to go.",
    "i need go home":           "I need to go home.",
    "i go home":                "I'm going home.",
    "i go school":              "I'm going to school.",
    "i go work":                "I'm going to work.",
    "i want":                   "I want.",
    "i want eat":               "I want to eat.",
    "i want drink":             "I want to drink.",
    "i want help":              "I want help.",
    "i want go":                "I want to go.",
    "i want go home":           "I want to go home.",
    "i want water":             "I want water.",
    "i want more":              "I want more.",
    "i want learn":             "I want to learn.",
    "i no understand":          "I don't understand.",
    "i not understand":         "I don't understand.",
    "i no know":                "I don't know.",
    "i don't know":             "I don't know.",
    "don't_know":               "I don't know.",
    "i no like":                "I don't like it.",
    "i not like":               "I don't like it.",
    "i no want":                "I don't want it.",

    # ── Questions ────────────────────────────────────────────────────
    "what":                     "What?",
    "what you want":            "What do you want?",
    "what you name":            "What is your name?",
    "what your name":           "What is your name?",
    "what you do":              "What are you doing?",
    "what you think":           "What do you think?",
    "what you need":            "What do you need?",
    "what you eat":             "What are you eating?",
    "what this":                "What is this?",
    "what that":                "What is that?",
    "what time":                "What time is it?",
    "what happen":              "What happened?",
    "where":                    "Where?",
    "where you":                "Where are you?",
    "where you go":             "Where are you going?",
    "where you live":           "Where do you live?",
    "where school":             "Where is the school?",
    "where home":               "Where is home?",
    "where water":              "Where is water?",
    "where friend":             "Where is my friend?",
    "who":                      "Who?",
    "who you":                  "Who are you?",
    "who that":                 "Who is that?",
    "who come":                 "Who is coming?",
    "who help":                 "Who will help?",
    "you understand":           "Do you understand?",
    "you want":                 "Do you want?",
    "you want eat":             "Do you want to eat?",
    "you want help":            "Do you want help?",
    "you hungry":               "Are you hungry?",
    "you ok":                   "Are you okay?",
    "you good":                 "Are you good?",
    "you ready":                "Are you ready?",
    "you know":                 "Do you know?",
    "you like":                 "Do you like it?",
    "you come":                 "Are you coming?",
    "you go":                   "Are you going?",
    "you need help":            "Do you need help?",

    # ── Commands / Requests ──────────────────────────────────────────
    "help":                     "Help!",
    "help me":                  "Help me!",
    "help please":              "Please help me!",
    "stop":                     "Stop!",
    "stop please":              "Stop, please!",
    "wait":                     "Wait.",
    "wait please":              "Wait, please.",
    "wait me":                  "Wait for me.",
    "come":                     "Come!",
    "come here":                "Come here!",
    "come please":              "Please come.",
    "go":                       "Go.",
    "go home":                  "Go home.",
    "go away":                  "Go away.",
    "go school":                "Go to school.",
    "sit down":                 "Sit down.",
    "stand up":                 "Stand up.",
    "more":                     "More.",
    "more please":              "More, please.",
    "again":                    "Again.",
    "again please":             "Again, please.",

    # ── Social phrases ───────────────────────────────────────────────
    "yes":                      "Yes.",
    "no":                       "No.",
    "yes please":               "Yes, please.",
    "no thank you":             "No, thank you.",
    "my name":                  "My name is",
    "you beautiful":            "You are beautiful.",
    "you friend":               "You are my friend.",
    "we friend":                "We are friends.",
    "i love":                   "I love.",
    "love you":                 "I love you.",
    "good":                     "Good.",
    "bad":                      "Bad.",
    "thumbs_up":                "Great!",

    # ── Common multi-sign combos ─────────────────────────────────────
    "think good":               "I think it's good.",
    "think bad":                "I think it's bad.",
    "i think good":             "I think it's good.",
    "i think bad":              "I think it's bad.",
    "know where":               "I know where it is.",
    "know what":                "I know what it is.",
    "no no":                    "No, no!",
    "yes yes":                  "Yes, yes!",
    "hello friend":             "Hello, friend!",
    "hello family":             "Hello, family!",
    "water please":             "Water, please.",
    "eat more":                 "Eat more.",
    "drink water":              "Drink water.",
    "go home now":              "I'm going home now.",
    "come home":                "Come home.",
    "school good":              "School is good.",
    "work good":                "Work is good.",
    "family good":              "My family is good.",
    "friend good":              "My friend is good.",
    "need more":                "I need more.",
    "want more":                "I want more.",
    "like school":              "I like school.",
    "like work":                "I like work.",
    "like friend":              "I like my friend.",
    "think you good":           "I think you're good.",
    "sorry wait":               "Sorry for the wait.",
    "thank you friend":         "Thank you, friend.",
    "stop wait":                "Stop and wait.",
}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PATTERN-BASED SENTENCE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════
# Regex patterns on the lowercased word sequence → template functions.
# Checked AFTER exact match fails but BEFORE spaCy corrections.

_QUESTION_WORDS = {"what", "where", "who", "how", "when", "why", "which"}
_SUBJECTS = {"i", "you", "he", "she", "we", "they", "it"}
_VERBS = {
    "go", "come", "eat", "drink", "want", "need", "like", "love",
    "help", "stop", "wait", "think", "know", "work", "learn",
    "see", "hear", "feel", "make", "give", "take", "say", "tell",
}
_ADJECTIVES = {
    "good", "bad", "happy", "sad", "tired", "sick", "hungry", "thirsty",
    "angry", "scared", "beautiful", "big", "small", "hot", "cold",
    "fast", "slow", "old", "young", "new", "ready", "busy", "free",
    "sorry", "late", "early", "right", "wrong", "nice", "fine", "ok",
}
_NOUNS = {
    "home", "school", "work", "water", "food", "friend", "family",
    "store", "hospital", "restaurant", "car", "bus", "phone", "book",
    "door", "room", "park", "doctor", "teacher", "baby", "child",
    "man", "woman", "boy", "girl", "name", "time", "help", "love",
    "money", "house",
}
_NEGATION = {"no", "not", "don't", "don't_know"}

# Subject → copula
_COPULA = {
    "i": "am", "you": "are", "we": "are", "they": "are",
    "he": "is", "she": "is", "it": "is",
}


def _pattern_compose(words: List[str]) -> Optional[str]:
    """
    Try to compose a natural sentence from sign words using structural patterns.
    Returns None if no pattern matches.
    """
    if not words:
        return None

    n = len(words)
    w0 = words[0]

    # ── Pattern: ends with question word → rearrange FIRST ────────────
    # Must be checked before subject patterns to catch "you name what"
    if n >= 2 and words[-1] in _QUESTION_WORDS:
        qw = words[-1]
        rest = words[:-1]
        # "you name what" → "What is your name?"
        if rest[0] in _SUBJECTS:
            subj = _fix_pronoun(rest[0])
            remainder = " ".join(rest[1:]) if len(rest) > 1 else ""
            if remainder:
                # Use "is" for questions about a single attribute (name, time, etc.)
                return f"{qw.capitalize()} is {_possessive(subj)} {remainder}?"
            copula = _COPULA.get(rest[0], "is")
            return f"{qw.capitalize()} {copula} {subj}?"
        return f"{qw.capitalize()} {' '.join(rest)}?"

    # ── Pattern: QUESTION_WORD ... → rearrange into English question ──
    if w0 in _QUESTION_WORDS:
        if n == 1:
            return w0.capitalize() + "?"
        rest = words[1:]
        # "what you name" → "What is your name?"
        # "where you go"  → "Where are you going?"
        subj = rest[0] if rest and rest[0] in _SUBJECTS else None
        if subj:
            copula = _COPULA.get(subj, "is")
            remainder = rest[1:]
            if not remainder:
                return f"{w0.capitalize()} {copula} {_fix_pronoun(subj)}?"
            verb_or_rest = " ".join(remainder)
            # Question + subject + verb
            if remainder[0] in _VERBS:
                verb = remainder[0]
                obj = " ".join(remainder[1:])
                if verb in ("go", "come"):
                    return f"{w0.capitalize()} {copula} {_fix_pronoun(subj)} {_progressive(verb)}{_prep_obj(obj)}?"
                return f"{w0.capitalize()} do {_fix_pronoun(subj)} {verb}{_prep_obj(obj)}?"
            # Question + subject + adjective
            if remainder[0] in _ADJECTIVES:
                return f"{w0.capitalize()} {copula} {_fix_pronoun(subj)} {' '.join(remainder)}?"
            # Question + subject + noun
            return f"{w0.capitalize()} {copula} {_fix_pronoun(subj)} {verb_or_rest}?"
        else:
            return f"{w0.capitalize()} {' '.join(rest)}?"

    # ── Pattern: SUBJECT + ADJECTIVE → "Subject is adjective." ────────
    if n >= 2 and w0 in _SUBJECTS and words[1] in _ADJECTIVES:
        subj = _fix_pronoun(w0)
        copula = _COPULA.get(w0, "is")
        adj = " ".join(words[1:])
        # Negation check
        if n >= 3 and words[1] in _NEGATION:
            adj = " ".join(words[2:])
            if w0 == "i":
                return f"I'm not {adj}."
            return f"{subj} {copula} not {adj}."
        if w0 == "i":
            return f"I'm {adj}."
        return f"{subj} {copula} {adj}."

    # ── Pattern: SUBJECT + NEGATION + VERB → "Subject don't verb." ────
    if n >= 3 and w0 in _SUBJECTS and words[1] in _NEGATION:
        subj = _fix_pronoun(w0)
        rest = words[2:]
        if rest[0] in _VERBS:
            neg = "don't" if w0 in ("i", "you", "we", "they") else "doesn't"
            return f"{subj} {neg} {' '.join(rest)}."
        copula = _COPULA.get(w0, "is")
        return f"{subj} {copula} not {' '.join(rest)}."

    # ── Pattern: SUBJECT + VERB + ... → build proper sentence ─────────
    if n >= 2 and w0 in _SUBJECTS:
        subj = _fix_pronoun(w0)
        verb = words[1]
        obj_words = words[2:]
        obj = " ".join(obj_words) if obj_words else ""

        if verb in _VERBS:
            # "want" + verb → "want to verb"
            if verb == "want" and obj_words and obj_words[0] in _VERBS:
                return f"{subj} want to {' '.join(obj_words)}."
            # "need" + verb → "need to verb"
            if verb == "need" and obj_words and obj_words[0] in _VERBS:
                return f"{subj} need to {' '.join(obj_words)}."
            # "go" / "come" + place → "going to place"
            if verb in ("go", "come") and obj_words:
                first_obj = obj_words[0]
                if first_obj in _NOUNS:
                    return f"{subj} {'am' if w0 == 'i' else _COPULA.get(w0, 'is')} {_progressive(verb)} to {obj}."
                return f"{subj} {'am' if w0 == 'i' else _COPULA.get(w0, 'is')} {_progressive(verb)} {obj}."
            # General: "I eat" → "I'm eating"
            if not obj_words:
                return f"{subj} {'am' if w0 == 'i' else _COPULA.get(w0, 'is')} {_progressive(verb)}."
            return f"{subj} {verb} {obj}."

        # Subject + noun → "Subject is at/a noun"
        if verb in _NOUNS:
            return f"{subj} {_COPULA.get(w0, 'is')} at {' '.join(words[1:])}."

    # ── Pattern: VERB alone or VERB + object → imperative ─────────────
    if w0 in _VERBS:
        if n == 1:
            return w0.capitalize() + "!"
        rest = " ".join(words[1:])
        if words[-1] == "please":
            core = " ".join(words[:-1])
            return f"Please {core}."
        return f"{w0.capitalize()} {rest}."

    # ── Pattern: ADJECTIVE alone → statement ──────────────────────────
    if w0 in _ADJECTIVES and n == 1:
        return w0.capitalize() + "."

    return None


def _fix_pronoun(word: str) -> str:
    """Capitalize I, leave others lowercase."""
    return "I" if word == "i" else word


def _possessive(subj: str) -> str:
    """Convert subject to possessive form."""
    table = {"I": "my", "you": "your", "he": "his", "she": "her",
             "we": "our", "they": "their", "it": "its"}
    return table.get(subj, subj + "'s")


def _progressive(verb: str) -> str:
    """Convert verb to present progressive (ing form)."""
    irregulars = {
        "eat": "eating", "drink": "drinking", "go": "going",
        "come": "coming", "make": "making", "give": "giving",
        "take": "taking", "have": "having", "run": "running",
        "sit": "sitting", "get": "getting", "swim": "swimming",
    }
    if verb in irregulars:
        return irregulars[verb]
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    if len(verb) >= 3 and verb[-1] not in "aeiouw" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _prep_obj(obj: str) -> str:
    """Add preposition before object if needed."""
    if not obj:
        return ""
    obj = obj.strip()
    if obj in _NOUNS:
        return f" to {obj}"
    return f" {obj}"


# ═══════════════════════════════════════════════════════════════════════════
# 3.  WORD-LEVEL CORRECTIONS (spaCy & basic)
# ═══════════════════════════════════════════════════════════════════════════

# Words where "to" is needed before a following noun/verb
_DIRECTIONAL_VERBS = {"go", "come", "walk", "run", "drive", "travel", "move", "want"}

# Common nouns that benefit from articles
_COUNTABLE_NOUNS = {
    "store", "house", "school", "hospital", "restaurant", "office",
    "car", "bus", "train", "book", "phone", "computer", "door",
    "table", "chair", "dog", "cat", "man", "woman", "boy", "girl",
    "child", "baby", "doctor", "teacher", "friend", "room", "park",
    "market", "shop", "street", "building", "city", "town",
}


def correct_grammar(raw_sentence: str) -> str:
    """
    Convert a raw sign-word sentence into natural English.
    Uses a layered approach: exact match → pattern templates → POS corrections.
    """
    if not raw_sentence or not raw_sentence.strip():
        return raw_sentence

    # Normalize: lowercase, collapse spaces, clean underscores
    text = raw_sentence.strip()
    text = text.replace("_", " ")
    lower = " ".join(text.lower().split())  # collapse whitespace

    # Layer 1: Exact phrase match
    if lower in _PHRASE_MAP:
        return _PHRASE_MAP[lower]

    # Layer 2: Prefix phrase match (e.g., "my name John" → "My name is John")
    for pattern, replacement in _PHRASE_MAP.items():
        if lower.startswith(pattern + " "):
            rest = text[len(pattern):].strip()
            result = replacement.rstrip(".!?") + " " + rest
            return _finalize(result)

    # Layer 3: Pattern-based sentence composition
    words = lower.split()
    composed = _pattern_compose(words)
    if composed is not None:
        return composed

    # Layer 4: Word-level corrections (spaCy or basic)
    if _SPACY_OK and _nlp is not None:
        text = _spacy_correct(text)
    else:
        text = _basic_correct(text)

    return _finalize(text)


def _finalize(text: str) -> str:
    """Capitalize first letter, ensure ending punctuation."""
    text = text.strip()
    if not text:
        return text
    # Fix double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    # Capitalize
    text = text[0].upper() + text[1:]
    # Fix "i " → "I "
    text = text.replace(" i ", " I ").replace(" i'", " I'")
    if text.startswith("i "):
        text = "I " + text[2:]
    # Add period if missing punctuation
    if text[-1] not in ".!?,;:":
        text += "."
    return text


def _spacy_correct(text: str) -> str:
    """Use spaCy POS tagging for smarter corrections."""
    doc = _nlp(text.lower())
    tokens = list(doc)
    result = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        prev = tokens[i - 1] if i > 0 else None
        nxt = tokens[i + 1] if i < len(tokens) - 1 else None
        word = tok.text

        # ── Rule: Insert copula before adjective after pronoun/noun ──
        if (tok.pos_ in ("ADJ",) or word in _ADJECTIVES):
            if prev and prev.pos_ in ("PRON", "NOUN", "PROPN"):
                # Check no copula already present
                if not (len(result) > 0 and result[-1] in ("am", "is", "are", "was", "were")):
                    subj = prev.text.lower()
                    copula = _COPULA.get(subj, "is")
                    result.append(copula)

        # ── Rule: Add article before bare countable noun ─────────────
        if tok.pos_ == "NOUN" and word in _COUNTABLE_NOUNS:
            if prev is None or prev.pos_ not in ("DET", "PRON", "ADJ", "NUM", "PROPN"):
                # Don't add if previous word is a preposition ("to")
                if not (len(result) > 0 and result[-1] in ("the", "a", "an", "to")):
                    result.append("the")

        # ── Rule: "want" + verb → "want to" + verb ──────────────────
        if word == "want" and nxt and nxt.pos_ == "VERB":
            result.append("want")
            result.append("to")
            i += 1
            continue

        # ── Rule: directional verb + noun → verb + "to" + noun ──────
        if (tok.pos_ == "NOUN" and prev and
                prev.lemma_ in _DIRECTIONAL_VERBS and prev.lemma_ != "want"):
            if not (len(result) > 0 and result[-1] == "to"):
                result.append("to")

        # ── Fix "I" ─────────────────────────────────────────────────
        if word == "i":
            word = "I"

        result.append(word)
        i += 1

    return " ".join(result)


def _basic_correct(text: str) -> str:
    """Simple rule-based correction without spaCy."""
    words = text.split()
    result = []

    for i, word in enumerate(words):
        w = word.lower()
        prev_w = words[i - 1].lower() if i > 0 else None
        nxt_w = words[i + 1].lower() if i < len(words) - 1 else None

        # Fix "I"
        if w == "i":
            word = "I"

        # Insert copula: pronoun + adjective
        if w in _ADJECTIVES and prev_w and prev_w in _COPULA:
            if not (len(result) > 0 and result[-1].lower() in ("am", "is", "are")):
                result.append(_COPULA[prev_w])

        # Add "to" after directional verb before noun
        if w in _COUNTABLE_NOUNS and prev_w in _DIRECTIONAL_VERBS:
            if not (len(result) > 0 and result[-1].lower() == "to"):
                result.append("to")
                result.append("the")

        result.append(word)

    return " ".join(result)


def is_available() -> bool:
    """Whether grammar correction is available (spaCy loaded)."""
    return True  # basic correction always works; spaCy enhances it
