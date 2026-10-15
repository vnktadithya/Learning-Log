import numpy as np

def tokenize(sentence: str)-> list[str]:
    return sentence.lower().split()

def build_vocab(pairs)-> dict:   # No test data is involved in building vocabulary so that is prevents any data leakage from test data and dont let the model to cheat by viewing at the data in prior 
    vocab = {'<unk>': 0}
    index = 1
    for pair in pairs:
        s1_vocab = tokenize(pair[0])
        s2_vocab = tokenize(pair[1])

        for word in s1_vocab + s2_vocab:
            if word not in vocab:
                vocab[word] = index
                index += 1

    return vocab

def BOW_vector(sentence: str, vocab: dict):
    tokens = tokenize(sentence)
    vector = [0] * len(vocab)

    for token in tokens:
        if token in vocab:
            index = vocab[token]
            vector[index] += 1

    return np.array(vector)

def to_ids(sentence: str, vocab: dict):
    tokens = tokenize(sentence)
    ids = []

    for token in tokens:
        if token not in vocab:
            ids.append(0)

        else:
            ids.append(vocab[token])

    return ids

def cosine_similarity(u, v):
    if np.count_nonzero(u) == 0 or np.count_nonzero(v) == 0:
        return 0

    dot_prod = np.dot(u, v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    cos_sim = dot_prod / (u_norm * v_norm)

    return cos_sim

def get_labels(pairs):
    labels = []
    for row in pairs:
        labels.append(row[2])

    return labels

def baseline_scores(pairs, vocab)-> list[float]:
    scores = []

    for index, row in enumerate(pairs):
        u = BOW_vector(row[0], vocab)
        v = BOW_vector(row[1], vocab)
        cos_sim = cosine_similarity(u, v)
        scores.append(cos_sim)

    return scores


def print_score_table(pairs, split, baseline, untrained) -> None:
    print("split    category    label      baseline cosine     untrained encoder cosine")
    scores = []

    for row, baseline_score, untrained_score in zip(pairs, baseline, untrained):
        print(f"{split}       {row[3]}         {row[2]}       {baseline_score}         {untrained_score}")


def auc(scores, labels):
    positive_pairs, negative_pairs = [], []
    positive_score, negative_score = 0, 0

    for index, label in enumerate(labels):
        if label == 1.0:
            positive_pairs.append(scores[index])
        else:
            negative_pairs.append(scores[index])

    for i in positive_pairs:
        for j in negative_pairs:
            if i > j:
                positive_score += 1
            elif i == j:
                positive_score += 0.5
                negative_score += 0.5
            else:
                negative_score += 1

    return positive_score / (positive_score + negative_score)

