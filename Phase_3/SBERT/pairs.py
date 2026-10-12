# Toy bi-encoder dataset: (sentence_a, sentence_b, label)
# label 1.0 = same root cause, 0.0 = different root cause
# Sentences are lowercase, no punctuation (tokenizer is whitespace split).
# "idx N" = example_index in Phase_2/LoRA/logs/ft20_model_outputs.jsonl (ground_truth only).

TRAIN_PAIRS = [
    # --- A: high overlap, same cause (1.0) ---
    ("valueerror because split on equals had no maxsplit so values with equals broke unpacking",   # idx 2
     "valueerror because the field was split on every equals sign instead of only the first", 1.0, 'A'),  # idx 2
    ("attributeerror because redis returned str keys and the code still called decode",             # idx 9
     "redis returned keys as str not bytes so calling decode raised attributeerror", 1.0, 'A'),          # idx 9
    ("dtype mismatch because tensors were created as float32 while the model ran in fp16",          # idx 18
     "tensors hardcoded to float32 caused a dtype mismatch when the model ran in half precision", 1.0, 'A'),  # idx 18

    # --- B: high overlap, different cause (0.0) ---
    ("attributeerror because redis returned str keys and the code still called decode",             # idx 9
     "attributeerror because the code called get partial shape on the wrapper tensor not the node", 0.0, 'B'),  # idx 32
    ("attributeerror because a wrong object was passed as the flow run argument",                   # idx 3
     "attributeerror because a non callback object was passed to model fit", 0.0, 'B'),                  # idx 34
    ("typeerror in hash because json dumps cannot serialize bytes attribute values",                # idx 29
     "typeerror in hash because the key class does not define a hash method", 0.0, 'B'),                 # idx 50

    # --- C: no overlap, same cause (1.0) — Note: not from the file ---
    ("pod restarts repeatedly with exit code 137",
     "kernel oom killer terminated process exceeding memory limit", 1.0, 'C'),
    ("connection refused on port 5432",
     "postgres server is not running", 1.0, 'C'),
    ("ssl certificate verify failed for internal api",
     "server presents a self signed cert missing from trust store", 1.0, 'C'),

    # --- D: low overlap, different cause (0.0) ---
    ("uppercase work pool name violated lowercase block name validation",                          # idx 4
     "cold dlopen could not resolve versioned rocm shared library", 0.0, 'D'),                           # idx 30
    ("gloo coordination across thousands of ranks caused socket timeouts",                          # idx 35
     "optional converter called a modern converter with one argument", 0.0, 'D'),                        # idx 40
    ("swallowed async exceptions left node attributes empty downstream",                            # idx 10
     "django serializer ran both ipv4 and ipv6 validators on input", 0.0, 'D'),                          # idx 43
]

# Held-out: new phrasings of causes seen in training, built from training vocab.
# Limitation: with 12 training pairs this tests paraphrase robustness within
# known causes, NOT generalization to unseen causes.
TEST_PAIRS = [
    # A
    ("the code called decode on str keys returned from redis",                                        # idx 9
     "redis keys were str so decode raised attributeerror", 1.0, 'A'),                           # idx 9
    # B
    ("attributeerror because the wrong object was passed to model fit",                             # idx 34
     "attributeerror because the wrong object was passed as the flow run", 0.0, 'B'),                    # idx 3
    # C
    ("pod restarts with exit code 137",
     "process exceeding memory limit", 1.0, 'C'),
    # D
    ("redis keys returned as str",                                                                  # idx 9
     "uppercase work pool name violated validation", 0.0, 'D'),                                          # idx 4
]