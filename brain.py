# =============================================================================
# brain.py — Shapatar Bot
# =============================================================================
#
# All bot logic lives here. Platform-agnostic.
# Takes strings in, returns strings out. Knows nothing about Discord,
# Telegram, or any interface.
#
# Pipeline stages:
#   1. preprocess()         — clean raw input
#   2. detect_intent()      — classify via embedding similarity
#   3. update_mood()        — FSM state transition
#   4. generate_response()  — assemble response from data pools
#   5. apply_style()        — surface personality transforms
#   6. process_message()    — master function, runs all stages
#
# =============================================================================

import re
import random

# Embedding classifier — replaces all rule-based intent detection
from embeddings import classify_intent, initialise as initialise_embeddings

# Response pools and constants from data.py
# Trigger lists no longer imported — embedding model handles classification
from data import (
    # Fence-sit assembly
    FENCE_SIT, FENCE_SIT_DOUBLE, DIRECT_YES, TOPIC_DODGE,

    # Response pools by intent
    FOLLOWUP_DEFLECTION, EAGER_VOLUNTEER, VOLUNTEER_FOLLOWUP,
    GREETING_RESPONSES, OPINION_RESPONSES,
    CARS_RESPONSES, IPHONE_RESPONSES, YOUNG_STUNNERS_RESPONSES,
    KARACHI_RESPONSES, TRAVEL_RESPONSES,
    GENERAL_CHAT,

    # Response pools by mood
    TENSE_FENCE_SIT, TENSE_GENERAL, TENSE_STRESS_ACKNOWLEDGE,
    SULKING_RESPONSES,
    EXPLOSION_OPENERS, EXPLOSION_MIDDLES, EXPLOSION_CLOSERS,
    EXPLOSION_AFTERMATH,

    # Mood transition constants
    STRESS_COUNT_TO_TENSE, STRESS_COUNT_TO_EXPLODE,
    SULK_RECOVERY_TURNS, TENSE_RECOVERY_TURNS,
    EXPLOSION_COOLDOWN_TURNS, EXPLOSION_PROBABILITY,

    # Invitation behaviour probabilities
    FENCE_SIT_PROBABILITY, DIRECT_YES_PROBABILITY,
    TOPIC_DODGE_PROBABILITY, DOUBLE_FENCE_SIT_PROB,

    # Style layer probabilities
    YAAR_INSERTION_PROB, ELLIPSIS_PROB, ABEY_PREFIX_PROB,
)


# =============================================================================
# MODEL INITIALISATION
# =============================================================================
#
# Runs when brain.py is first imported — before the conversation loop.
# Model download (~1.1GB) happens here on first run, cached afterward.
# Subsequent startups load the cached index in ~0.04s.
# =============================================================================

print("[Shapatar Bot] Starting up — loading semantic model...")
initialise_embeddings()
print("[Shapatar Bot] Ready.\n")


# =============================================================================
# PREPROCESSING
# =============================================================================

# Compiled once at module load — reused for every message
PUNCTUATION_PATTERN = re.compile(r'[!؟?.,،۔]+')


def preprocess(text: str) -> str:
    """
    Clean raw user input into normalised form for intent classification.

    Pure function: string in, string out, no side effects.

    Transformations:
    1. Lowercase        — case-insensitive matching
    2. Strip edges      — remove leading/trailing whitespace
    3. Remove punctuation — "hain!" matches "hain"
    4. Collapse spaces  — "chal  chalte" matches "chal chalte"
    """
    text = text.lower()
    text = text.strip()
    text = PUNCTUATION_PATTERN.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# INTENT DETECTION
# =============================================================================

def detect_intent(cleaned_text: str, last_intent: str = None) -> tuple:
    """
    Classify cleaned input into an intent category using semantic embeddings.

    Thin wrapper around classify_intent() from embeddings.py.
    Interface identical to v1 rule-based version — returns (intent, negated).

    The embedding model handles negation implicitly: "nahi chalte hain"
    does not match invitation because the negative phrasing shifts the
    vector away from the invitation cluster. negated always returns False.

    Args:
        cleaned_text:  Preprocessed message from preprocess().
        last_intent:   Previous turn's intent — used by generate_response()
                       for conversational context, not by classifier.

    Returns:
        (intent: str, negated: bool)
    """
    if len(cleaned_text.strip()) <= 2:
        return "general", False

    intent = classify_intent(cleaned_text)
    return intent, False


# =============================================================================
# STATE MACHINE — MOOD MANAGEMENT
# =============================================================================

def create_context() -> dict:
    """
    Create and return a fresh conversation context dictionary.

    Called once at conversation start in main.py.
    Passed into process_message() on every turn and returned updated.
    """
    return {
        "mood":               "normal",
        "stress_count":       0,
        "turns_in_mood":      0,
        "explosion_cooldown": 0,
        "total_turns":        0,
        "last_intent":        None,
        "last_response":      None,
    }


def update_mood(context: dict, intent: str) -> dict:
    """
    Apply FSM transition logic: given context and detected intent,
    return updated context reflecting any mood change.

    Does NOT generate a response — mood management is one job,
    response generation is a separate job.

    Works on a copy of context — original is not mutated.
    """
    ctx = context.copy()

    ctx["total_turns"]        += 1
    ctx["turns_in_mood"]      += 1
    ctx["explosion_cooldown"]  = max(0, ctx["explosion_cooldown"] - 1)
    ctx["last_intent"]         = intent

    current_mood = ctx["mood"]

    # ---- TRANSITIONS FROM: normal ----------------------------------------
    if current_mood == "normal":
        if intent == "anger":
            ctx = _transition_to(ctx, "sulking")
        elif intent == "stress":
            ctx["stress_count"] += 1
            if ctx["stress_count"] >= STRESS_COUNT_TO_TENSE:
                ctx = _transition_to(ctx, "tense")
                ctx["stress_count"] = 0
        else:
            # Stress count decays naturally during calm conversation
            ctx["stress_count"] = max(0, ctx["stress_count"] - 1)

    # ---- TRANSITIONS FROM: tense -----------------------------------------
    elif current_mood == "tense":
        if intent == "anger":
            ctx = _transition_to(ctx, "sulking")
        elif intent == "stress":
            ctx["stress_count"] += 1
            if ctx["stress_count"] >= STRESS_COUNT_TO_EXPLODE:
                if (ctx["explosion_cooldown"] == 0 and
                        random.random() < EXPLOSION_PROBABILITY):
                    ctx = _transition_to(ctx, "exploding")
                    ctx["explosion_cooldown"] = EXPLOSION_COOLDOWN_TURNS
                else:
                    ctx["stress_count"] = 0
        else:
            if ctx["turns_in_mood"] >= TENSE_RECOVERY_TURNS:
                ctx = _transition_to(ctx, "normal")

    # ---- TRANSITIONS FROM: sulking ---------------------------------------
    elif current_mood == "sulking":
        if ctx["turns_in_mood"] >= SULK_RECOVERY_TURNS:
            ctx = _transition_to(ctx, "normal")
        if intent == "anger":
            # Reminding him of what made him sulk resets the timer
            ctx["turns_in_mood"] = 0

    # ---- TRANSITIONS FROM: exploding -------------------------------------
    elif current_mood == "exploding":
        # Explosion is a single-turn state — immediately transitions to sulking
        ctx = _transition_to(ctx, "sulking")

    return ctx


def _transition_to(ctx: dict, new_mood: str) -> dict:
    """
    Perform a mood transition, resetting counters.

    Extracted as helper — the same reset logic applies to every transition
    regardless of which one triggered it. Avoids copy-pasting reset logic.
    """
    ctx["mood"]         = new_mood
    ctx["turns_in_mood"] = 0
    ctx["stress_count"]  = 0
    return ctx


# =============================================================================
# RESPONSE GENERATION
# =============================================================================

def generate_response(intent: str, context: dict) -> str:
    """
    Select or assemble a response based on detected intent and current mood.

    Knows WHAT to say. Does not know HOW he says things — that is
    apply_style()'s job.
    """
    mood = context["mood"]

    # Guard 1: Exploding — overrides intent entirely
    if mood == "exploding":
        opener = random.choice(EXPLOSION_OPENERS)
        middle = random.choice(EXPLOSION_MIDDLES)
        closer = random.choice(EXPLOSION_CLOSERS)
        return f"{opener}\n{middle}\n{closer}"

    # Guard 2: Sulking — minimal cold responses regardless of intent
    if mood == "sulking":
        return random.choice(SULKING_RESPONSES)

    # Tense mood — still processes intent but with tense-specific pools
    if mood == "tense":
        return _generate_tense_response(intent, context)

    # Normal mood — full personality
    return _generate_normal_response(intent, context)


def _generate_tense_response(intent: str, context: dict) -> str:
    """Generate response for tense mood state."""
    if intent == "invitation":
        return random.choice(TENSE_FENCE_SIT)
    if intent == "stress":
        return random.choice(TENSE_STRESS_ACKNOWLEDGE)
    if intent in ("cars", "iphone", "young_stunners", "karachi", "travel"):
        return random.choice(_get_passion_pool(intent))
    return random.choice(TENSE_GENERAL)


def _generate_normal_response(intent: str, context: dict) -> str:
    """Generate response for normal mood state — full personality active."""

    # ---- INVITATION: the fence-sit ----------------------------------------
    if intent == "invitation":

        # Follow-up on previous fence-sit → double fence-sit
        if context["last_intent"] == "invitation":
            if random.random() < DOUBLE_FENCE_SIT_PROB:
                return random.choice(FENCE_SIT_DOUBLE)

        # Weighted choice between three invitation behaviours
        behaviour = random.choices(
            ["fence_sit", "direct_yes", "topic_dodge"],
            weights=[FENCE_SIT_PROBABILITY,
                     DIRECT_YES_PROBABILITY,
                     TOPIC_DODGE_PROBABILITY],
            k=1
        )[0]

        if behaviour == "direct_yes":
            return random.choice(DIRECT_YES)
        if behaviour == "topic_dodge":
            return random.choice(TOPIC_DODGE)

        # Assemble fence-sit from parts — combinatorial variety
        agreement   = random.choice(FENCE_SIT["agreements"])
        reassurance = random.choice(FENCE_SIT["reassurances"])
        pivot       = random.choice(FENCE_SIT["pivots"])
        objection   = random.choice(FENCE_SIT["objections"])
        return f"{agreement}, {reassurance}. {pivot} {objection}."

    # ---- FOLLOW-UP: pressing him on what he said --------------------------
    if intent == "followup":
        if context["last_intent"] in ("invitation", "followup"):
            return random.choice(FOLLOWUP_DEFLECTION)
        return random.choice(GENERAL_CHAT)

    # ---- HELP REQUEST: eager volunteering ---------------------------------
    if intent == "help_request":
        if context["last_intent"] == "help_request":
            return random.choice(VOLUNTEER_FOLLOWUP)
        return random.choice(EAGER_VOLUNTEER)

    # ---- GREETING ---------------------------------------------------------
    if intent == "greeting":
        return random.choice(GREETING_RESPONSES)

    # ---- OPINION REQUEST --------------------------------------------------
    if intent == "opinion":
        return random.choice(OPINION_RESPONSES)

    # ---- PASSION TOPICS ---------------------------------------------------
    if intent in ("cars", "iphone", "young_stunners", "karachi", "travel"):
        return random.choice(_get_passion_pool(intent))

    # ---- GENERAL FALLBACK -------------------------------------------------
    return random.choice(GENERAL_CHAT)


def _get_passion_pool(intent: str) -> list:
    """Return response pool for a passion topic intent."""
    pools = {
        "cars":           CARS_RESPONSES,
        "iphone":         IPHONE_RESPONSES,
        "young_stunners": YOUNG_STUNNERS_RESPONSES,
        "karachi":        KARACHI_RESPONSES,
        "travel":         TRAVEL_RESPONSES,
    }
    return pools.get(intent, GENERAL_CHAT)


# =============================================================================
# STYLE LAYER
# =============================================================================

def apply_style(response: str, mood: str) -> str:
    """
    Apply personality-wide surface transforms to any response.

    Knows HOW he says things. Does not know what was said.

    Style layer is completely skipped for sulking and exploding:
    the absence of his normal warmth IS the emotional signal.
    Adding "yaar" to a sulking response would destroy the characterisation.
    """
    if mood in ("sulking", "exploding"):
        return response

    # Transform 1: ABEY prefix (normal mood only — not tense)
    if mood == "normal" and random.random() < ABEY_PREFIX_PROB:
        if not response.lower().startswith("abey"):
            response = "abey " + response

    # Transform 2: YAAR suffix (avoid doubling up)
    if random.random() < YAAR_INSERTION_PROB:
        if "yaar" not in response.lower():
            response = response + " yaar"

    # Transform 3: ELLIPSIS suffix (trailing vagueness)
    if random.random() < ELLIPSIS_PROB:
        if not response.endswith("..."):
            response = response + "..."

    return response


# =============================================================================
# MASTER PIPELINE FUNCTION
# =============================================================================

def process_message(user_text: str, context: dict) -> tuple:
    """
    Single public entry point for the entire bot pipeline.

    The only function main.py needs to call.

    Stages run in this exact order — order matters:
    1. preprocess()          — clean input
    2. detect_intent()       — classify intent
    3. update_mood()         — transition FSM BEFORE generating response
    4. generate_response()   — select/assemble raw content
    5. apply_style()         — apply surface transforms

    Mood is updated before response generation so the response reflects
    the NEW mood, not the old one.

    Args:
        user_text:  Raw string from the user.
        context:    Current conversation context from create_context()
                    or previous call to process_message().

    Returns:
        (response: str, updated_context: dict)
    """
    cleaned  = preprocess(user_text)
    intent, _ = detect_intent(cleaned, context.get("last_intent"))
    context  = update_mood(context, intent)
    raw      = generate_response(intent, context)
    final    = apply_style(raw, context["mood"])

    context["last_response"] = final
    return final, context
