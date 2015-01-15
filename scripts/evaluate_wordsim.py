# Author: Karl Stratos (karlstratos@gmail.com)
"""
This *Python Version 2* module is used to evaluate given word embeddings on the
word similarity task. We use Python 2 (not 3) to utilize the scipy functions
that calculate Pearson's and Spearman's rank correlations.

Argument 1 [input 1: each line = word1 word2 similarity_score]
Argument 2 [input 2: word embeddings file]
"""
import sys
from numpy import array
from numpy import dot
from numpy import linalg
from scipy import stats

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
    embedding = {}
    dim = 0
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                word = tokens[1]
                if word in similarity_vocab:
                    # We need this embedding.
                    values = []
                    for i in range(2, len(tokens)):
                        values.append(float(tokens[i]))
                    dim = len(values)
                    embedding[word] = array(values)
                    embedding[word] /= linalg.norm(embedding[word])
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

    pearson = stats.pearsonr(x, y)[0]
    print "Pearson's correlation coefficient: {0:.3f}".format(pearson)

    spearman = stats.spearmanr(x, y)[0]
    print "Spearman's rank correlation coefficient: {0:.3f}".format(spearman)

if __name__ == "__main__":
    # Path to word similarity file.
    SIMILARITY_PATH = sys.argv[1]

    # Path to word embeddings file.
    EMBEDDING_PATH = sys.argv[2]

    evaluate_wordsim(SIMILARITY_PATH, EMBEDDING_PATH)
