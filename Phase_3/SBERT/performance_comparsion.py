from pairs import TRAIN_PAIRS, TEST_PAIRS
from encoder import Encoder, encoder_scores
from preprocessing import build_vocab, baseline_scores, get_labels, print_score_table, auc
import torch

def performance_comparision():
    vocab = build_vocab(TRAIN_PAIRS)
    torch.manual_seed(2)
    encoder = Encoder(len(vocab))
    # For Training pairs
    baseline_train_scores = baseline_scores(TRAIN_PAIRS, vocab)
    encoder_train_scores = encoder_scores(TRAIN_PAIRS, encoder, vocab)
    train_labels = get_labels(TRAIN_PAIRS)
    print_score_table(TRAIN_PAIRS, 'Train', baseline_train_scores, encoder_train_scores)
    print("----------------------------------------------------------")
    print(f"Training pairs AUC score based on baseline cosine: {auc(baseline_train_scores, train_labels)}")
    print("----------------------------------------------------------")
    print(f"Training pairs AUC score based on untrained encoder cosine: {auc(encoder_train_scores, train_labels)}")
    print("----------------------------------------------------------")


    # For Test pairs
    baseline_test_scores = baseline_scores(TEST_PAIRS, vocab)
    encoder_test_scores = encoder_scores(TEST_PAIRS, encoder, vocab)
    test_labels = get_labels(TEST_PAIRS)
    print_score_table(TEST_PAIRS, 'Test', baseline_test_scores, encoder_test_scores)
    print("----------------------------------------------------------")
    print(f"Testing pairs AUC score based on baseline cosine: {auc(baseline_test_scores, test_labels)}")
    print("----------------------------------------------------------")
    print(f"Testing pairs AUC score based on untrained encoder cosine: {auc(encoder_test_scores, test_labels)}")
    print("----------------------------------------------------------")


if __name__ == '__main__':
    performance_comparision()