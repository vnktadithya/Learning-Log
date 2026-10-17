import torch
import torch.nn.functional as F
from torch import optim
from preprocessing import to_ids, build_vocab, baseline_scores
from encoder import Encoder
from pairs import TRAIN_PAIRS

def batch_loss(encoder, pairs, vocab) -> torch.Tensor:
    losses = []
    for row in pairs:
        sent_1_ids = torch.tensor(to_ids(row[0], vocab), dtype=torch.long)
        sent_2_ids = torch.tensor(to_ids(row[1], vocab), dtype=torch.long)
        u = encoder(sent_1_ids)
        v = encoder(sent_2_ids)
        cos_sim = F.cosine_similarity(u, v, dim=0)
        losses.append((cos_sim - row[2])**2)
    return torch.stack(losses).mean()

def train_encoder(encoder, pairs, vocab, steps=300, lr=0.01) -> list[float]:
    optimizer = optim.Adam(encoder.parameters(), lr = lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()   #gradients should be reset to 0 at the beggining of every step so that gradients won't add on for the next steps. 
        loss = batch_loss(encoder, pairs, vocab)    #compute batch loss
        loss.backward()     #  backpropagate
        optimizer.step()    #  update the weights
        history.append(loss.item())
        if step % 25 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    return history

if __name__ == '__main__':
    vocab = build_vocab(TRAIN_PAIRS)
    torch.manual_seed(0)
    encoder = Encoder(len(vocab))
    train_batch_loss = batch_loss(encoder, TRAIN_PAIRS, vocab)
    print(f'Train batch loss: {train_batch_loss.item()}')
    print(f'{train_batch_loss.requires_grad}')
    print(f'{train_batch_loss.grad_fn}')

    baseline_cosine_scores = baseline_scores(TRAIN_PAIRS, vocab)
    baseline_cosine_losses = []
    for score, row in zip(baseline_cosine_scores, TRAIN_PAIRS):
        baseline_cosine_losses.append((score - row[2])**2)

    print(baseline_cosine_losses)
    print(f'Baseline mean loss: {sum(baseline_cosine_losses)/len(baseline_cosine_losses)}')