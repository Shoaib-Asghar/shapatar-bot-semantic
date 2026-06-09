# =============================================================================
# embeddings.py — Shapatar Bot
# =============================================================================
#
# Handles everything related to the embedding model:
#   - Loading multilingual-e5-base
#   - Building the intent index (centroid vectors per intent)
#   - Caching the index to disk (rebuilt automatically when examples change)
#   - Computing cosine similarity between a new message and all intents
#   - Returning the best-matching intent above a confidence threshold
#
# DEPENDENCY:
#   pip install sentence-transformers
#
# MODEL: intfloat/multilingual-e5-base
#   ~1.1GB, downloads automatically on first use.
#   Set HF_HOME in .env to control download location.
#   No GPU required. Runs on CPU (~80-200ms per message on i5-8th gen).
#
# E5 REQUIREMENT:
#   This model requires prompt prefixes.
#   Queries (incoming messages)  → prepend "query: "
#   Passages (example sentences) → prepend "passage: "
#   Without these prefixes, similarity scores degrade noticeably.
#
# CACHE BEHAVIOUR:
#   intent_index.pkl is built once and saved to disk.
#   On subsequent startups it loads in ~0.04s instead of ~20s.
#   If intent_examples.py is modified, the cache rebuilds automatically.
#   intent_index.pkl is listed in .gitignore — never commit it.
# =============================================================================

import os
import pickle
import time
import numpy as np

# Set threading env vars before torch is imported to prevent segfault
# on Windows + Intel CPU combinations
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from dotenv import load_dotenv
load_dotenv()

# Redirect model download location if HF_HOME is set in .env
hf_home = os.getenv('HF_HOME')
if hf_home:
    os.environ['HF_HOME'] = hf_home

from sentence_transformers import SentenceTransformer
from intent_examples import INTENT_EXAMPLES


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME           = "intfloat/multilingual-e5-base"
INDEX_CACHE_FILE     = "intent_index.pkl"
EXAMPLES_FILE        = "intent_examples.py"

# Cosine similarity must exceed this value for an intent to be returned.
# Below this threshold the classifier returns "general".
#
# Tuning guide:
#   Too high (e.g. 0.70) → too many messages fall through to general
#   Too low  (e.g. 0.25) → wrong intents match confidently
#   Sweet spot for Roman Urdu: 0.40-0.50
#   Adjust based on debug_similarities() output during testing.
SIMILARITY_THRESHOLD = 0.42


# =============================================================================
# MODULE-LEVEL STATE
# =============================================================================
# Loaded once, reused for every call. Loading is expensive (~2-3s).
# Using is fast (~80-200ms per message).

_model        = None   # Loaded SentenceTransformer model
_intent_index = None   # Dict: intent name → centroid vector (numpy array)


# =============================================================================
# INITIALISATION
# =============================================================================

def initialise():
    """
    Load the embedding model and build (or load cached) intent index.

    Call this once at bot startup before the conversation loop begins.
    Subsequent calls are no-ops — module-level state already populated.
    """
    global _model, _intent_index

    if _model is not None and _intent_index is not None:
        return

    print(f"[Embeddings] Loading model: {MODEL_NAME}")
    print("[Embeddings] First run downloads ~1.1GB. Subsequent runs use cache.")

    start  = time.time()
    _model = SentenceTransformer(MODEL_NAME)
    elapsed = time.time() - start
    print(f"[Embeddings] Model loaded in {elapsed:.1f}s")

    _intent_index = _load_or_build_index()
    print(f"[Embeddings] Ready. {len(_intent_index)} intents indexed.")


def _load_or_build_index() -> dict:
    """
    Load the intent index from disk if fresh, otherwise build and save it.

    Freshness check: cache is stale if intent_examples.py was modified
    more recently than intent_index.pkl. Editing examples automatically
    triggers a rebuild on next startup — no manual cache deletion needed.
    """
    if _is_cache_fresh():
        print("[Embeddings] Loading index from cache...")
        start = time.time()
        with open(INDEX_CACHE_FILE, "rb") as f:
            index = pickle.load(f)
        elapsed = time.time() - start
        print(f"[Embeddings] Index loaded from cache in {elapsed:.2f}s")
        return index

    print("[Embeddings] Building intent index from examples...")
    print("[Embeddings] (Takes ~10-20s on first run, then cached)")
    start = time.time()
    index = _build_index()
    elapsed = time.time() - start
    print(f"[Embeddings] Index built in {elapsed:.1f}s")
    _save_cache(index)
    return index


def _is_cache_fresh() -> bool:
    """
    Return True if cache exists and is newer than intent_examples.py.

    Uses file modification timestamps (mtime) — the same principle
    as the Make build system: only rebuild what has changed.
    """
    if not os.path.exists(INDEX_CACHE_FILE):
        return False
    cache_mtime    = os.path.getmtime(INDEX_CACHE_FILE)
    examples_mtime = os.path.getmtime(EXAMPLES_FILE)
    return cache_mtime > examples_mtime


def _build_index() -> dict:
    """
    Encode all example sentences and compute per-intent centroid vectors.

    For each intent:
    1. Prepend "passage: " to all examples (E5 requirement)
    2. Encode in one batch — faster than one at a time
    3. Average the vectors → one centroid per intent
    4. Normalise to unit length → enables dot product as cosine similarity

    Returns:
        Dict: { intent_name: centroid_vector (numpy array, shape 768) }
    """
    index = {}

    for intent_name, examples in INTENT_EXAMPLES.items():
        if not examples:
            print(f"[Embeddings] Warning: no examples for '{intent_name}', skipping")
            continue

        # E5 requires "passage: " prefix for indexed/stored text
        prefixed = [f"passage: {ex}" for ex in examples]

        # Encode full list in one batch.
        # normalize_embeddings=True → unit vectors → dot product = cosine sim
        vectors = _model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        # Centroid: average across all example vectors
        # np.mean(axis=0) collapses (num_examples, 768) → (768,)
        centroid = np.mean(vectors, axis=0)

        # Re-normalise after averaging — averaging unit vectors
        # does not produce a unit vector
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        index[intent_name] = centroid
        print(f"[Embeddings]   ✓ {intent_name} ({len(examples)} examples)")

    return index


def _save_cache(index: dict):
    """Serialise index to disk using pickle."""
    with open(INDEX_CACHE_FILE, "wb") as f:
        pickle.dump(index, f)
    print(f"[Embeddings] Index cached to {INDEX_CACHE_FILE}")


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_intent(text: str) -> str:
    """
    Classify an incoming message into one of the known intents.

    This is the function brain.py calls — the drop-in replacement
    for all rule-based detect_intent() logic.

    Process:
    1. Prepend "query: " to message (E5 requirement for queries)
    2. Encode message → 768-dim unit vector
    3. Dot product against every centroid (= cosine similarity)
    4. Return intent with highest similarity if above SIMILARITY_THRESHOLD
    5. Return "general" if nothing clears the threshold

    Args:
        text: Preprocessed message string from preprocess() in brain.py.

    Returns:
        Intent string matching a key in INTENT_EXAMPLES, or "general".
    """
    if _model is None or _intent_index is None:
        initialise()

    # E5 requirement: "query: " prefix for the text being searched
    query        = f"query: {text}"
    query_vector = _model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Compute dot product (= cosine sim) against every intent centroid
    similarities = {
        intent: float(np.dot(query_vector, centroid))
        for intent, centroid in _intent_index.items()
    }

    best_intent = max(similarities, key=lambda k: similarities[k])
    best_score  = similarities[best_intent]

    if best_score >= SIMILARITY_THRESHOLD:
        return best_intent

    return "general"


def debug_similarities(text: str) -> dict:
    """
    Return similarity scores for all intents for a given text.

    Use this from the terminal when tuning the threshold or
    debugging unexpected classifications:

        from embeddings import initialise, debug_similarities
        initialise()
        scores = debug_similarities("nikal na yaar")
        for intent, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = '█' * int(score * 40)
            print(f'{score:.3f}  {intent:20s} {bar}')

    This shows exactly how the model scored every intent, telling you
    whether the threshold needs adjusting or examples need improving.
    """
    if _model is None or _intent_index is None:
        initialise()

    query        = f"query: {text}"
    query_vector = _model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return {
        intent: float(np.dot(query_vector, centroid))
        for intent, centroid in _intent_index.items()
    }
