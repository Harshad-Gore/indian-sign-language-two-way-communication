"""
NLP Engine — converts English text into ISL-friendly gloss sequences.

ISL grammar rules applied:
  1. Tokenise the sentence
  2. POS tag every token
  3. Remove stopwords / auxiliary verbs
  4. Reorder to Subject-Object-Verb (SOV) pattern
  5. Uppercase remaining tokens (gloss convention)

Example:
  Input:  "What is your name?"
  Output: NLPBreakdown(gloss_sequence=["YOUR", "NAME", "WHAT"], ...)
"""

from __future__ import annotations
import re
import shutil
from pathlib import Path
from typing import Optional
from loguru import logger

# ── NLTK bootstrap ───────────────────────────────────────────────────────────

_NLTK_DATA_DIR = Path(__file__).resolve().parents[1] / "nltk_data"


def _ensure_nltk():
    import nltk

    _NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = str(_NLTK_DATA_DIR)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)

    def _download(pkg: str) -> None:
        try:
            nltk.download(pkg, download_dir=data_dir, quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download failed for {pkg}: {e}")

    def _has_punkt() -> bool:
        base = _NLTK_DATA_DIR / "tokenizers" / "punkt"
        if not base.exists():
            return False
        if (base / "PY3_tab").exists():
            return True
        if (base / "PY3" / "english.pickle").exists():
            return True
        if (base / "english.pickle").exists():
            return True
        return False

    if not _has_punkt():
        shutil.rmtree(_NLTK_DATA_DIR / "tokenizers" / "punkt", ignore_errors=True)
        _download("punkt")

    for rel_path, pkg in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]:
        if not (_NLTK_DATA_DIR / rel_path).exists():
            _download(pkg)


_spacy_nlp = None

def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except Exception as e:
            logger.warning(f"spaCy unavailable ({e}), using NLTK fallback")
    return _spacy_nlp


# ── ISL grammar constants ─────────────────────────────────────────────────────

AUX_VERBS = {
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "have", "has", "had",
}

FUNCTION_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "into", "through", "during", "before",
    "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "that", "which", "who", "whom", "whose", "where", "when",
    "how", "this", "these", "those", "such", "as",
}

QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which"}

# POS tags to KEEP after filtering
KEEP_POS = {
    "NN", "NNS", "NNP", "NNPS",   # Nouns
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs (main)
    "JJ", "JJR", "JJS",           # Adjectives
    "RB", "RBR", "RBS",           # Adverbs
    "PRP", "PRP$", "WP", "WP$",   # Pronouns
    "CD",                          # Cardinal number
    "WRB",                         # WH-adverb (where, when, why, how)
    "WDT",                         # WH-determiner (what, which)
}


# ── Main NLP pipeline ─────────────────────────────────────────────────────────

def process_text(text: str, apply_isl_grammar: bool = True) -> dict:
    """
    Full NLP pipeline returning structured breakdown + gloss sequence.
    """
    _ensure_nltk()
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    text_clean = text.strip()
    is_question = text_clean.endswith("?")

    # Tokenise
    tokens_raw = word_tokenize(text_clean)

    # POS tag
    tagged = nltk.pos_tag(tokens_raw)

    # ── Try spaCy for richer analysis ──────────────────────────────────────
    spacy_doc = None
    nlp = _get_spacy()
    if nlp:
        try:
            spacy_doc = nlp(text_clean)
        except Exception:
            pass

    # ── Build token analysis list ──────────────────────────────────────────
    token_analyses = []
    kept_tokens = []

    # Resolve stopwords set
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set()

    for word, pos in tagged:
        word_lower = word.lower()
        is_punct = not any(c.isalpha() for c in word)
        is_aux = word_lower in AUX_VERBS
        is_func = word_lower in FUNCTION_WORDS
        is_stop = word_lower in stop_words
        is_question_word = word_lower in QUESTION_WORDS

        keep = (
            not is_punct
            and pos in KEEP_POS
            and not is_aux
            and not is_func
        ) or is_question_word

        if apply_isl_grammar:
            kept = keep
        else:
            kept = not is_punct

        # Get spaCy details if available
        dep = "—"
        lemma = word
        if spacy_doc:
            for tok in spacy_doc:
                if tok.text.lower() == word_lower:
                    dep = tok.dep_
                    lemma = tok.lemma_
                    break

        token_analyses.append({
            "token": word,
            "pos": pos,
            "dep": dep,
            "lemma": lemma,
            "is_stopword": is_stop or is_aux or is_func,
            "kept_in_isl": kept,
        })

        if kept:
            kept_tokens.append((word, pos, is_question_word))

    # ── SOV Reorder ────────────────────────────────────────────────────────
    if apply_isl_grammar:
        gloss_sequence = _reorder_sov(kept_tokens, is_question)
    else:
        gloss_sequence = [w.upper() for w, _, _ in kept_tokens if w.isalpha()]

    simplified = " ".join(gloss_sequence)

    # ── Sentence structure label ───────────────────────────────────────────
    if is_question:
        structure = "INTERROGATIVE (WH-question)" if any(
            w.lower() in QUESTION_WORDS for w, _, _ in kept_tokens
        ) else "YES/NO QUESTION"
    elif any(pos.startswith("V") for _, pos, _ in kept_tokens):
        structure = "DECLARATIVE (action)"
    else:
        structure = "NOMINAL"

    return {
        "original": text_clean,
        "simplified": simplified,
        "gloss_sequence": gloss_sequence,
        "tokens": token_analyses,
        "sentence_structure": structure,
    }


def _reorder_sov(tokens: list[tuple], is_question: bool) -> list[str]:
    """
    Apply ISL Subject-Object-Verb reordering.
    Question words go to the END in ISL.
    """
    subjects, objects, verbs, adjectives, adverbs, question_words, others = (
        [], [], [], [], [], [], []
    )

    for word, pos, is_qword in tokens:
        w = word.upper()
        if is_qword:
            question_words.append(w)
        elif pos.startswith("PRP") or pos == "WP":
            subjects.append(w)
        elif pos in {"NN", "NNS", "NNP", "NNPS"}:
            objects.append(w)
        elif pos.startswith("VB"):
            verbs.append(w)
        elif pos.startswith("JJ"):
            adjectives.append(w)
        elif pos.startswith("RB"):
            adverbs.append(w)
        else:
            others.append(w)

    # ISL order: Subject + Adjectives + Object + Adverbs + Verb + Question
    ordered = subjects + adjectives + objects + adverbs + verbs + others + question_words
    return [w for w in ordered if w.isalpha() or any(c.isalnum() for c in w)]
