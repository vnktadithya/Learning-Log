from pairs import TRAIN_PAIRS, TEST_PAIRS
from encoder import Encoder, encoder_scores
from preprocessing import build_vocab, baseline_scores, get_labels, print_score_table, auc
from train import train_encoder, batch_loss
import torch

def performance_comparision():
    vocab = build_vocab(TRAIN_PAIRS)
    for seed in [0,1,2]: 
        print(f"-----------------------------------------------------------------------------------------------")
        print(f"                                        SEED: {seed}                                           ")
        print(f"-----------------------------------------------------------------------------------------------")
        torch.manual_seed(seed)
        encoder = Encoder(len(vocab))

        # Baseline scores
        baseline_train_scores = baseline_scores(TRAIN_PAIRS, vocab)
        baseline_test_scores = baseline_scores(TEST_PAIRS, vocab)

        # Untrained encoder scores
        untrained_encoder_train_scores = encoder_scores(TRAIN_PAIRS, encoder, vocab)
        untrained_encoder_test_scores = encoder_scores(TEST_PAIRS, encoder, vocab)

        # Trained encoder scores
        trained_encoder_loss_history = train_encoder(encoder, TRAIN_PAIRS, vocab)   # this steps trains the encoder and updates the parmeters in place
        trained_encoder_train_scores = encoder_scores(TRAIN_PAIRS, encoder, vocab)
        trained_encoder_test_scores = encoder_scores(TEST_PAIRS, encoder, vocab)

        # Get Train and Test labels
        train_labels = get_labels(TRAIN_PAIRS)
        test_labels = get_labels(TEST_PAIRS)

        # Print score table on training pairs and compute AUC for each model
        print_score_table(TRAIN_PAIRS, 'Train', baseline_train_scores, untrained_encoder_train_scores, trained_encoder_train_scores)
        print("----------------------------------------------------------")
        print(f"Training pairs AUC score based on baseline cosine: {auc(baseline_train_scores, train_labels)}")
        print("----------------------------------------------------------")
        print(f"Training pairs AUC score based on untrained encoder cosine(seed: {seed}): {auc(untrained_encoder_train_scores, train_labels)}")
        print("----------------------------------------------------------")
        print(f"Training pairs AUC score based on trained encoder cosine(seed: {seed}): {auc(trained_encoder_train_scores, train_labels)}")
        print("----------------------------------------------------------")


        # Print score table on testing pairs and compute AUC for each model
        print_score_table(TEST_PAIRS, 'Test', baseline_test_scores, untrained_encoder_test_scores, trained_encoder_test_scores)
        print("----------------------------------------------------------")
        print(f"Testing pairs AUC score based on baseline cosine: {auc(baseline_test_scores, test_labels)}")
        print("----------------------------------------------------------")
        print(f"Testing pairs AUC score based on untrained encoder cosine(seed: {seed}): {auc(untrained_encoder_test_scores, test_labels)}")
        print("----------------------------------------------------------")
        print(f"Testing pairs AUC score based on trained encoder cosine(seed: {seed}): {auc(trained_encoder_test_scores, test_labels)}")
        print("----------------------------------------------------------")

        with torch.no_grad():
            final_train_loss = batch_loss(encoder, TRAIN_PAIRS, vocab).item()
            test_loss = batch_loss(encoder, TEST_PAIRS, vocab).item()
        print(f"Final MSE for trained Encoder(seed: {seed}):")
        print(f"On training set: {final_train_loss}")
        print(f"On testing set: {test_loss}")

        print('\n')


if __name__ == '__main__':
    performance_comparision()