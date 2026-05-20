"""
intent_predictor.py  —  Load the trained pipeline and predict intents.

Two-layer prediction:
  1. Keyword fallback  — catches short / ambiguous queries that confuse the
     model (e.g. "where is my class" without "now").
  2. ML model          — TF-IDF + LogisticRegression pipeline trained in
     nlp_model.py.

If the keyword layer fires with high confidence it overrides the ML result.
"""

import pickle
import re
import logging
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH               = Path("model.pkl")
DEFAULT_CONFIDENCE_THRESHOLD = 0.30   # lowered; keyword layer handles edge cases


# ----------------------------
# LOAD MODEL
# ----------------------------
def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Run nlp_model.py first to train and save the model."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


_pipeline = _load(MODEL_PATH)


# ----------------------------
# RESULT STRUCTURE
# ----------------------------
@dataclass
class IntentResult:
    intent:       str
    confidence:   float
    is_confident: bool


# ----------------------------
# PREPROCESS
# ----------------------------
@lru_cache(maxsize=1000)
def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


# ----------------------------
# KEYWORD FALLBACK
# Maps keyword sets → intent.  Evaluated in priority order; first match wins.
# Each rule: (required_words, forbidden_words, intent)
# A rule fires when ALL required_words appear AND no forbidden_words appear.
# ----------------------------
_KEYWORD_RULES = [
    # teacher / professor / sir / mam  (check before class rules)
    ({"teacher"},                           set(),          "teacher_location"),
    ({"professor"},                         set(),          "teacher_location"),
    ({"faculty"},                           set(),          "teacher_location"),
    ({"sir"},                               {"class"},      "teacher_location"),
    ({"mam"},                               {"class"},      "teacher_location"),

    # next class  (must check before current_class)
    ({"next", "class"},                     set(),          "next_class"),
    ({"next", "lecture"},                   set(),          "next_class"),
    ({"next", "subject"},                   set(),          "next_class"),
    ({"next", "period"},                    set(),          "next_class"),
    ({"upcoming", "class"},                 set(),          "next_class"),
    ({"upcoming", "lecture"},               set(),          "next_class"),
    ({"after", "class"},                    set(),          "next_class"),

    # current class
    ({"class"},                             {"next","today","timetable","schedule","all","list"}, "current_class"),
    ({"lecture"},                           {"next","today","timetable","schedule","all","list"}, "current_class"),
    ({"subject"},                           {"next","today","timetable","schedule","all","list","fee"}, "current_class"),
    ({"period"},                            {"next","today","timetable","schedule","fee"},        "current_class"),

    # today's schedule
    ({"timetable"},                         set(),          "todays_schedule"),
    ({"schedule"},                          set(),          "todays_schedule"),
    ({"today", "class"},                    set(),          "todays_schedule"),
    ({"today", "lecture"},                  set(),          "todays_schedule"),
    ({"classes", "today"},                  set(),          "todays_schedule"),

    # fee
    ({"fee"},                               set(),          "fee_info"),
    ({"fees"},                              set(),          "fee_info"),
    ({"tuition"},                           set(),          "fee_info"),
    ({"pay"},                               set(),          "fee_info"),

    # greeting
    ({"hello"},                             set(),          "greeting"),
    ({"hi"},                                set(),          "greeting"),
    ({"hey"},                               set(),          "greeting"),
    ({"morning"},                           set(),          "greeting"),
    ({"afternoon"},                         set(),          "greeting"),
    ({"evening"},                           set(),          "greeting"),
    ({"namaste"},                           set(),          "greeting"),

    # thanks
    ({"thanks"},                            set(),          "thanks"),
    ({"thank"},                             set(),          "thanks"),
    ({"appreciate"},                        set(),          "thanks"),
    ({"cheers"},                            set(),          "thanks"),
]


def _keyword_predict(words: set) -> str | None:
    """Return an intent name if a keyword rule fires, else None."""
    for required, forbidden, intent in _KEYWORD_RULES:
        if required.issubset(words) and not forbidden.intersection(words):
            return intent
    return None


# ----------------------------
# PREDICT  (MAIN)
# ----------------------------
def predict(text: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> IntentResult:
    processed = preprocess(text)

    if not processed:
        return IntentResult("unknown", 0.0, False)

    word_set = set(processed.split())

    # ── Layer 1: keyword fallback ────────────────────────────────────────
    kw_intent = _keyword_predict(word_set)

    # ── Layer 2: ML model ────────────────────────────────────────────────
    try:
        proba      = _pipeline.predict_proba([processed])[0]
        ml_conf    = float(proba.max())
        ml_intent  = _pipeline.classes_[proba.argmax()]
    except AttributeError:
        ml_intent  = _pipeline.predict([processed])[0]
        ml_conf    = 0.5
    except Exception as exc:
        logger.warning("Prediction error: %s", exc)
        ml_intent, ml_conf = "unknown", 0.0

    logger.debug(
        "text=%r  kw=%s  ml=%s(%.2f)", text, kw_intent, ml_intent, ml_conf
    )

    # ── Decision logic ───────────────────────────────────────────────────
    # If ML is confident, trust it (it has seen the full sentence context).
    # If ML is uncertain, fall back to the keyword rule.
    if ml_conf >= confidence_threshold:
        final_intent = ml_intent
        final_conf   = ml_conf
    elif kw_intent is not None:
        # keyword override — assign a nominal confidence so is_confident=True
        final_intent = kw_intent
        final_conf   = 0.75
    else:
        final_intent = "unknown"
        final_conf   = ml_conf

    return IntentResult(
        intent       = final_intent,
        confidence   = round(final_conf, 4),
        is_confident = final_intent != "unknown",
    )


# ----------------------------
# BACKWARD COMPATIBILITY
# ----------------------------
def get_intent(text: str, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> IntentResult:
    """Alias for predict() kept for backward compatibility."""
    return predict(text, confidence_threshold)