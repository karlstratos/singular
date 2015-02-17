# Author: Karl Stratos (karlstratos@gmail.com)
"""
This module is used to evaluate word embeddings on the word similarity task.
"""
import argparse
from numpy import array
from numpy import dot
from numpy import linalg

def read_normalized_embeddings(embedding_path, no_counts, ignore_line1):
    """Reads normalized embeddings."""
    embedding = {}
    dim = 0
    line1_ignored = False
    with open(embedding_path, "r") as embedding_file:
        for line in embedding_file:
            tokens = line.split()
            if len(tokens) > 0:
                if ignore_line1 and (not line1_ignored):
                    line1_ignored = True
                    continue
                if no_counts: # <word> <values...>
                    word = tokens[0]
                    starting_index = 1
                else: # <count> <word> <values...>
                    word = tokens[1]
                    starting_index = 2
                values = []
                for i in range(starting_index, len(tokens)):
                    values.append(float(tokens[i]))
                if dim:
                    assert(len(values) == dim)
                else:
                    dim = len(values)
                embedding[word] = array(values)
                embedding[word] /= linalg.norm(embedding[word])  # Make norm 1.
    return embedding, dim

def read_similarity_data(similarity_path):
    """Reads word pairs and similarity scores."""
    word_pairs = []
    human_scores = []
    vocab = {}
    with open(similarity_path, "r") as similarity_file:
        for line in similarity_file:
            tokens = line.split()
            if len(tokens) > 0:
                word1 = tokens[0]
                word2 = tokens[1]
                similarity_score = float(tokens[2])
                word_pairs.append((word1, word2))
                human_scores.append(similarity_score)
                vocab[word1] = True
                vocab[word2] = True
    return word_pairs, human_scores, vocab

def assign_embedding(embedding, vocab):
    """
    Assigns an embedding for each word in the vocabulary, if any.
    """
    assigned_embedding = {}
    for word in vocab:
        if word in embedding:
            # If we have the exact match in embeddings, just use it.
            assigned_embedding[word] = embedding[word]
        elif word.lower() in embedding:
            # If we do *not* have the exact match but a lowercased match in
            # embeddings, we will use that instead: e.g., "Microsoft" in
            # assigned_embedding might refers to "microsoft" in embedding.
            # This can happen when the corpus was lowercased.
            assigned_embedding[word] = embedding[word.lower()]

    return assigned_embedding

def transform_to_averaged_ranks(values):
    """Computes averaged ranks for the given values."""
    sorted_values = sorted(values)
    averaged_ranks = []
    index = 0
    while index < len(sorted_values):
        num_same = 1
        rank_sum = index + 1
        while index+1 < len(sorted_values) and \
              abs(sorted_values[index + 1] - sorted_values[index]) < 1e-15:
            index += 1
            num_same += 1
            rank_sum += index + 1

        averaged_rank = float(rank_sum) / num_same
        for i in range(num_same):
            averaged_ranks.append(averaged_rank)
        index += 1

    # Map each value to the corresponding index in averaged_ranks. A value can
    # appear many times but it doesn't matter since it will have the same
    # averaged rank.
    #
    # E.g.,
    #       values         = [     1.5,   0.7,  0.33,   0.7    ]
    #       sorted_values  = [    0.33,   0.7,   0.7,   1.5    ]
    #       averaged_ranks = [       1,   2.5,   2.5,     4    ]
    #
    # Either
    #       value2index    = {  0.33:0,        0.7:2, 1.5:3    }
    #       sorted_indices = [       3,     2,     0,     2    ]
    # or
    #       value2index    = {  0.33:0,   0.7:1,      1.5:3    }
    #       sorted_indices = [       3,     1,     0,     1    ]
    #
    # will lead to: transformed_values = [4, 2.5, 1, 2.5]
    value2index = {}
    for i, value in enumerate(sorted_values):
        value2index[value] = i
    rank_indices = [value2index[value] for value in values]

    transformed_values = []
    for i in rank_indices:
        transformed_values.append(averaged_ranks[i])

    return transformed_values

def compute_spearmans_correlation(values1, values2):
    """Computes Spearman's rank correlation coefficient."""
    assert(len(values1) == len(values2))

    values1_transformed = transform_to_averaged_ranks(values1)
    values2_transformed = transform_to_averaged_ranks(values2)

    num_instances = len(values1)
    sum_squares = sum([pow(values1_transformed[i] - values2_transformed[i], 2)
                       for i in range(num_instances)])
    uncorrelatedness = 6.0 * sum_squares / \
                       (num_instances * (pow(num_instances, 2) - 1))
    correlation = 1.0 - uncorrelatedness
    return correlation

def main(args):
    """
    Computes the cosine similarity between each pair of word vectors (if known).
    Then reports how this correlates to human judgment in Spearman's rank
    correlation coefficient.
    """
    embedding, dim = read_normalized_embeddings(args.embedding_path,
                                                args.no_counts,
                                                args.ignore_line1)
    print("{0} {1}-dimensional embeddings.".format(len(embedding), dim))

    word_pairs_all, human_scores_all, vocab = \
        read_similarity_data(args.similarity_path)
    print("{0} word pairs with similarity scores.".format(len(word_pairs_all)))

    assigned_embedding = assign_embedding(embedding, vocab)
    print("{0}/{1} word types have embeddings.".format(len(assigned_embedding),
                                                       len(vocab)))

    human_scores = []
    cosine_scores = []
    num_evaluated = 0
    for i, (word1, word2) in enumerate(word_pairs_all):
        if word1 in assigned_embedding and word2 in assigned_embedding:
            human_scores.append(human_scores_all[i])
            cosine_scores.append(dot(assigned_embedding[word1],
                                     assigned_embedding[word2]))
            num_evaluated += 1
    spearman_score = compute_spearmans_correlation(human_scores, cosine_scores)
    print("Spearman's correlation: {0:.3f} ({1}/{2} evaluated)".format(
            spearman_score, num_evaluated, len(word_pairs_all)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("similarity_path", type=str, help="path to word "
                           "similarity file.")
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file.")
    argparser.add_argument("--no_counts", action="store_true", help="Embeddings"
                           " don't counts for the first column?")
    argparser.add_argument("--ignore_line1", action="store_true", help="Ignore "
                           "the first line in the embeddings file?")
    parsed_args = argparser.parse_args()
    main(parsed_args)
