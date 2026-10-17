# OOV probe pairs: (sentence_a, sentence_b, label, group)
# Out-of-vocabulary is defined relative to build_vocab(TRAIN_PAIRS).
# Kept separate from pairs.py on purpose: this is a probe, not part of the
# main evaluation, so the TEST_PAIRS results stay comparable to earlier logs.
#
# POS-k / NEG-k: the same pair with progressively more words swapped for
# unseen synonyms (k = 0 is the in-vocab original). This isolates the effect
# of OOV words from the effect of the cause itself.
# FULL-*: every word in both sentences is out of vocabulary.

OOV_PAIRS = [
    # Same cause (idx 9: redis returned str keys, code called decode) - label 1.0
    ("the code called decode on str keys returned from redis",
     "redis keys were str so decode raised attributeerror", 1.0, 'POS-0'),
    ("the code invoked decode on str keys returned from redis",
     "redis keys were str so decode threw attributeerror", 1.0, 'POS-1'),
    ("the client invoked decode on string keys fetched from redis",
     "redis keys were strings so decode threw attributeerror", 1.0, 'POS-2'),
    ("the client invoked decoding on string entries fetched from the cache",
     "cache entries were strings so decoding threw an exception", 1.0, 'POS-3'),

    # Different cause (idx 9 vs idx 4: str keys vs uppercase pool name) - label 0.0
    ("redis keys returned as str",
     "uppercase work pool name violated validation", 0.0, 'NEG-0'),
    ("redis keys returned as strings",
     "uppercase work pool name violated naming rules", 0.0, 'NEG-1'),
    ("cache entries returned as strings",
     "capitalized queue label violated naming rules", 0.0, 'NEG-2'),
    ("stale cache entries came back stringified",
     "capitalized queue label rejected by naming rules", 0.0, 'FULL-NEG'),

    # Same cause, every word unseen - label 1.0
    ("jvm crashed after heap exhaustion",
     "garbage collector thrashing preceded crash", 1.0, 'FULL-POS'),
]