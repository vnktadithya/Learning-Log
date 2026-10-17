import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import build_vocab, to_ids, BOW_vector
from pairs import TRAIN_PAIRS

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        vector = self.embedding(ids)
        sentence_mean_pool_vector = vector.mean(dim= 0)

        return sentence_mean_pool_vector

    

def encoder_scores(pairs, encoder, vocab) -> list[float]:
    scores = []

    with torch.no_grad():
        for index, row in enumerate(pairs):
            sent_1_ids = torch.tensor(to_ids(row[0], vocab), dtype=torch.long)
            sent_2_ids = torch.tensor(to_ids(row[1], vocab), dtype=torch.long)
            u = encoder(sent_1_ids)
            v = encoder(sent_2_ids)
            cos_sim = F.cosine_similarity(u, v, dim=0)
            scores.append(cos_sim.item())

    return scores


if __name__ == '__main__':
    input_sentence = 'valueerror because split on equals had no maxsplit so values with equals broke unpacking'
    input_length = len(input_sentence.split())
    vocab = build_vocab(TRAIN_PAIRS)
    ids = torch.tensor(to_ids(input_sentence, vocab), dtype=torch.long)
    torch.manual_seed(0)
    encoder = Encoder(len(vocab))
    print(encoder)
    encoded_sentence = encoder(ids)
    bow_vector = BOW_vector(input_sentence, vocab)
    bow_tensor = torch.tensor(bow_vector, dtype = torch.float32)
    mean_bow_tensor = bow_tensor/input_length
    final_bow_tensor = (mean_bow_tensor) @ encoder.embedding.weight
    print(torch.allclose(encoded_sentence, final_bow_tensor))
    print(encoded_sentence.shape)
    print(encoded_sentence.requires_grad)
    print(encoded_sentence.grad_fn)