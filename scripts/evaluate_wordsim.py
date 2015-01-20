# Author: Karl Stratos (karlstratos@gmail.com)
"""
This module is used to evaluate given word embeddings on the word similarity
task. We use scipy functions to calculate Pearson's and Spearman's rank
correlations.

Argument 1 [input 1: each line = word1 word2 similarity_score]
Argument 2 [input 2: word embeddings file]
"""
import sys
from math import sqrt
from numpy import array
from numpy import argsort
from numpy import dot
from numpy import linalg
from scipy.stats.mstats import spearmanr

def read_normalized_embeddings(embedding_path):
    """Read embeddings from the given path."""
    embedding = {}
    dim = 0
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                word = tokens[1]
                values = []
                for i in range(2, len(tokens)):
                    values.append(float(tokens[i]))
                if dim:
                    assert len(values) == dim
                else:
                    dim = len(values)
                embedding[word] = array(values)
                embedding[word] /= linalg.norm(embedding[word])
    return embedding, dim

def spearmans_correlation(x, y):
    tol = 1e-10;
    assert len(x) == len(y)

    # Sort values in each.
    x_sorted = sorted(x)
    y_sorted = sorted(y)

    # Calculate ranks.
    x_ranks = []
    index = 0
    while index < len(x_sorted):
        num_same = 1
        rank_sum = index + 1
        while index+1 < len(x_sorted) and \
              abs(x_sorted[index+1] - x_sorted[index]) < tol:
            index += 1
            num_same += 1
            rank_sum += index + 1

        for i in range(num_same):
            x_ranks.append(float(rank_sum)/num_same)
        index += 1

    y_ranks = []
    index = 0
    while index < len(y_sorted):
        num_same = 1
        rank_sum = index + 1
        while index+1 < len(y_sorted) and \
              abs(y_sorted[index+1] - y_sorted[index]) < tol:
            index += 1
            num_same += 1
            rank_sum += index + 1

        for i in range(num_same):
            y_ranks.append(float(rank_sum)/num_same)
        index += 1

    # Figure out which value corresponds to which position.
    mydict1 = {}
    for i, x_val in enumerate(x_sorted):
        mydict1[x_val] = i  # May be many-to-one (doesn't matter).
    x_indices = [mydict1[x_val] for x_val in x]

    mydict2 = {}
    for i, y_val in enumerate(y_sorted):
        mydict2[y_val] = i  # May be many-to-one (doesn't matter).
    y_indices = [mydict2[y_val] for y_val in y]

    final_x = []
    for i in x_indices:
        final_x.append(x_ranks[i])
    final_y = []
    for i in y_indices:
        final_y.append(y_ranks[i])

    d_sum = 0
    for i in xrange(len(x_ranks)):
        d_sum += pow(final_x[i] - final_y[i], 2)
    n = len(x_ranks)
    correlation = 1.0 - (6.0 * d_sum) / (n * (pow(n, 2) - 1) )

    return correlation

def evaluate_wordsim(similarity_path, embedding_path):
    """
    Computes the cosine similarity between each pair of word vectors (if known).
    Then reports how this correlates to human judgment both in Pearson's
    correlation coefficient coefficient (linear) and Spearman's rank correlation
    coefficient (not necessarily linear).
    """
    # Read pairs of words and their (human) similarity scores.
    word_pairs = []
    human_scores = []
    similarity_vocab = {}
    with open(similarity_path, "r") as similarity_file:
        for line in similarity_file:
            tokens = line.split()
            if len(tokens) > 0:
                word1 = tokens[0]
                word2 = tokens[1]
                similarity_score = float(tokens[2])
                word_pairs.append((word1, word2))
                human_scores.append(similarity_score)
                similarity_vocab[word1] = True
                similarity_vocab[word2] = True
    print "Read {0} word pairs with similarity scores".format(len(word_pairs))

    # Read (and normalize) embeddings for word types we need.
    embedding, dim = read_normalized_embeddings(embedding_path)
    print "Read {0} embeddings of dimension {1}".format(len(embedding), dim)

    # Compute consine similarity scores based on the normalized embeddings.
    x = []
    y = []
    num_skipped = 0
    for i, (word1, word2) in enumerate(word_pairs):
        if word1 in embedding and word2 in embedding:
            x.append(human_scores[i])
            y.append(dot(embedding[word1], embedding[word2]))
        else:
            num_skipped += 1
    print "Skipped {0} pairs out of {1} that lack embeddings".format(
        num_skipped, len(word_pairs))

    spearman = spearmans_correlation(x, y)
    spearman_scipy = spearmanr(x, y, use_ties=False)[0]
    assert abs(spearman - spearman_scipy) < 1e-15
    print "Spearman's rank correlation coefficient: {0:.3f}".format(spearman)

if __name__ == "__main__":
    # Path to word similarity file.
    SIMILARITY_PATH = sys.argv[1]

    # Path to word embeddings file.
    EMBEDDING_PATH = sys.argv[2]

    evaluate_wordsim(SIMILARITY_PATH, EMBEDDING_PATH)
