"""OOV probe for the toy bi-encoder.

Retrains the seed-0 encoder exactly as performance_comparision.py does
(same vocab, seed, steps, lr), runs two integrity checks, then scores
OOV_PAIRS with both the BoW baseline and the trained encoder.
Results are printed and written to oov_probe.log (UTF-8).
"""
import torch
import os
from pairs import TRAIN_PAIRS
from oov_pairs import OOV_PAIRS
from preprocessing import build_vocab, tokenize, baseline_scores
from encoder import Encoder, encoder_scores
from train import train_encoder

SEED = 0
DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DIR, "logs", "oov_probe.log")


def oov_ratio(sentence: str, vocab: dict) -> str:
    tokens = tokenize(sentence)
    unseen = sum(1 for t in tokens if t not in vocab)
    return f"{unseen}/{len(tokens)}"


def main() -> None:
    lines = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    vocab = build_vocab(TRAIN_PAIRS)

    torch.manual_seed(SEED)
    encoder = Encoder(len(vocab))
    unk_before = encoder.embedding.weight[0].detach().clone()

    train_encoder(encoder, TRAIN_PAIRS, vocab)

    # Check: the <unk> row (index 0) never appears in a training sentence,
    # so it should receive no gradient and stay exactly at its initial value.
    unk_after = encoder.embedding.weight[0].detach()
    log(f"Seed {SEED} | vocab size {len(vocab)}")
    log(f"<unk> row unchanged by training: {torch.equal(unk_before, unk_after)}")
    log()

    base = baseline_scores(OOV_PAIRS, vocab)
    trained = encoder_scores(OOV_PAIRS, encoder, vocab)

    header = f"{'group':<10}{'label':>6}{'oov_a':>8}{'oov_b':>8}{'baseline':>11}{'trained':>11}"
    log(header)
    log("-" * len(header))
    for (a, b, label, group), bs, ts in zip(OOV_PAIRS, base, trained):
        log(f"{group:<10}{label:>6.1f}{oov_ratio(a, vocab):>8}{oov_ratio(b, vocab):>8}"
            f"{float(bs):>11.4f}{ts:>11.4f}")

    log()
    log("Reproducibility: POS-0 and NEG-0 are TEST_PAIRS A and D verbatim, so their")
    log("trained cosines must equal the seed-0 test column in performance_comparision.log.")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()