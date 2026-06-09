# Shapatar Bot v2 — Semantic Embedding Approach

This is the second iteration of Shapatar Bot, an attempt to improve on the
rule-based v1 by replacing its hardcoded intent detection with a proper
semantic embedding model. The goal was to understand natural Roman Urdu input
without requiring hand-written trigger lists for every possible phrasing.

The intent classification is objectively bad but the bot is still surprisingly
usable.
---

## Background

[Shapatar Bot v1](https://github.com/Shoaib-Asghar/shapatar-bot) is a
rule-based Python chatbot that mimics a specific person's Karachi conversational
style. It uses hardcoded trigger lists, compiled regex root patterns, a
proximity-based negation checker, and a finite state machine for mood tracking
across conversation turns. The personality(fence-sitting, eager volunteering,
sulking, occasional explosions of anger) lives entirely in
hand-authored data files.

v1 works for phrases it was explicitly programmed for. Its failure mode is
honest: if it doesn't recognise a phrase, it falls back to a generic
in-character response. The problem is that Roman Urdu has essentially infinite
ways to say the same thing, and maintaining trigger lists for all of them is
a losing battle. Someone typing `"kidhar nikalna hai phir aaj"` gets a generic
response because none of those words appear in the trigger list.

The obvious question was: can a semantic embedding model handle this?
---

## The Approach

Instead of matching phrases against hand-written lists, v2 encodes example
sentences for each intent into dense vectors using a pre-trained multilingual
embedding model. When a new message arrives, it gets encoded into the same
vector space and compared against each intent's centroid using cosine
similarity. The closest match above a confidence threshold wins.

The rest of the bot: the state machine, mood transitions, response pools,
style layer is identical to v1. Only the intent detection mechanism changed.

### Model Choice

We used `intfloat/multilingual-e5-base`, a multilingual sentence embedding
model from Microsoft. Actual download size: 1.1GB. It was selected because:

- It was trained with contrastive learning on sentence pairs, optimising
  directly for semantic similarity
- Its training data includes Common Crawl, which contains Pakistani internet
  content and therefore some exposure to Roman Urdu

### Architecture

```
shapatar_bot_v2/
├── intent_examples.py   — natural example sentences per intent
├── embeddings.py        — model loading, index building, cosine similarity
├── data.py              — response pools and constants
├── brain.py             — pipeline, state machine, responses
└── main.py              — terminal conversation loop
```

The intent index is built once at startup by encoding all example sentences,
averaging them into per-intent centroid vectors, and caching to disk. The
cache invalidates automatically if `intent_examples.py` is modified.

---

## Evaluation Results

Tested against 65 phrases across 13 intents, 5 per intent.

```
============================================================
EVALUATION REPORT
============================================================
Total Examples Tested : 65
Total Correct         : 30
Overall Accuracy      : 46.15%
Average Latency       : 210.3 ms per message

--- Accuracy by Intent ---
invitation     :   0.0% (0/5)
opinion        :  80.0% (4/5)
help_request   :  40.0% (2/5)
stress         :   0.0% (0/5)
anger          :  20.0% (1/5)
greeting       :  20.0% (1/5)
cars           : 100.0% (5/5)
iphone         :  80.0% (4/5)
young_stunners :  60.0% (3/5)
karachi        :  60.0% (3/5)
travel         : 100.0% (5/5)
followup       :  40.0% (2/5)
general        :   0.0% (0/5)
============================================================
```

The pattern in the scores explains why. Every single classification, correct
or wrong, scores between 0.83 and 0.89 in cosine similarity:

```
[X] 'kidhar nikalna hai phir aaj'  (should be: invitation)
      iphone     : 0.8683
      travel     : 0.8622
      invitation : 0.8621    ← correct answer is 0.0062 behind the winner

[X] 'bhai koi scene on karo yaar'  (should be: invitation)
      karachi    : 0.8772
      travel     : 0.8768
      cars       : 0.8767    ← invitation doesn't even appear in top 3
```

This is embedding anisotropy, all vectors collapse into a narrow cone so
every sentence looks roughly 86% similar to every other sentence. The
classification is essentially a coin flip between whoever happens to be 0.003
ahead on that particular sentence. The threshold we set does nothing useful
because every prediction clears it regardless of correctness.

Travel and cars score 100% not because the model understands them better, but
because their vocabulary is specific enough (`dubai`, `nathiagali`, `civic`,
`tyre pressure`) that the model can latch onto those words directly even without
real semantic understanding. The intents that fail — invitation, stress, general
— are expressed through short generic phrases where the model has nothing
distinctive to grab.

---

## Actual Conversation Output

Despite the numbers above, here is what the bot actually produces in
conversation:

```

You: trip pe jain
Bot: bhai road trip mein gaadi solid honi chahiye, nahi toh problem

You: manage karay ga
Bot: han dekhte hain, koi na koi raasta hoga...

You: karachi chalte hain
Bot: yaar Burns Road ka khaana kha ke dekho, duniya mein kuch nahi...

You: kaam complete kar pehle
Bot: bhai ho jayega, jugaar lagaonga...

You: kahin chalen
Bot: abey yaar main bata raha tha lekin woh bhi ek cheez hai...

```

This is noticeably better than the 46% accuracy number suggests. The
personality is consistent throughout. Most responses feel in-character even
when the intent classification is wrong.

---

## Why The Numbers Are Bad But The Experience Is Tolerable

This gap between metric and experience is worth understanding because it says
something real about how to evaluate chatbots.

**The response pools are designed to absorb misclassification.** The person
being mimicked is already vague and non-committal by design. Fence-sitting,
deflecting, saying "ho jayega" — these responses are in-character for almost
any input.

**Some misclassifications land on semantically adjacent intents.** `"karachi
chalte hain"` should be invitation. The model classified it as karachi. The
karachi response pool contains things about Burns Road and Sea View — which
is a perfectly reasonable response to being invited somewhere in Karachi. The
wrong label produced the right response. This is luck, but it happens often
enough to matter.

**The state machine works regardless of classification accuracy.** Mood
transitions, sulking, tension, these run on intent labels and are affected
by classification errors, but the mood system is fairly robust to noise.

**What actually breaks:** The behaviours that require specific intent
detectio, the fence-sit on invitations, the angry sulking, the escalation
chain from normal to tense to exploding, these depend on getting the right
label. A greeting classified as travel produces a coherent-sounding response
but not the right one. Stress signals that never register as stress mean
the mood machine never builds toward tension. The characterisation is
flattened rather than broken.

So the bot produces output that feels like a person talking, but it does not faithfully reproduce the specific behaviours. 

---

## What Went Wrong Technically

### Embedding Anisotropy

Multilingual models trained on diverse formal text tend to pack their vectors
into a narrow cone of the vector space rather than spreading them across all
dimensions. When this happens, cosine similarity between any two sentences is
uniformly high regardless of actual meaning. In a healthy embedding space,
unrelated sentences score around 0.2-0.5 similarity. Our scores are 0.83-0.89
for everything so we are not able to distinguish between different intents.

### Roman Urdu Is Not In The Training Data

`multilingual-e5-base` is multilingual, but its training data is formal written
text . Roman Urdu — Urdu transliterated into Latin script, code-switched with
English, no standardised spelling, developed in WhatsApp and SMS — is essentially
absent. When the model encounters `"scene on karo"` or `"sab kuch sar par aa
gaya hai"`, it does not recognise the semantics. It produces vectors based on
shallow surface features and the results are arbitrary.

### Short Casual Sentences Have Weak Signals

Even ignoring the above two problems, 4-7 word casual sentences in an informal
register are hard to discriminate by embedding similarity. They share too much
surface vocabulary and register for the centroid-based approach to find clean
separation. This is a fundamental limit of the method for this type of input.

---

## Running It

```bash
python main.py
```
Set a custom cache
location to avoid filling your C drive:

```
# In .env
HF_HOME=D:\your_preferred_path
```
