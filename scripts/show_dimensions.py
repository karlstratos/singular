# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to show the most prominent (in absolute magnitude) words
per dimension.
"""
import argparse
from numpy import array
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

def main(args):
    """Prints the most prominent (in absolute magnitude) words per dimension."""
    embedding, dim = read_normalized_embeddings(args.embedding_path,
                                                args.no_counts,
                                                args.ignore_line1)

    while True:
        try:
            i = int(input("Type a dimension (or just quit the program): "))
            if i >= dim:
                print("Given {0} - need <= {1}".format(i, dim-1))
            else:
                value_word_pairs = []
                for word in embedding:
                    value_word_pairs.append((embedding[word][i], word))
                value_word_pairs.sort(reverse=True)
                for j in range(20):
                    value, word = value_word_pairs[j]
                    print("\t\t{0:.4f}\t\t{1}".format(value, word))

        except (ValueError):
            print("Please input an integer in [0, {0}]".format(dim-1))
            continue

        except (KeyboardInterrupt, EOFError):
            print()
            exit(0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("embedding_path", type=str, help="path to word "
                           "embeddings file")
    argparser.add_argument("--no_counts", action="store_true", help="embeddings"
                           " don't counts for the first column?")
    argparser.add_argument("--ignore_line1", action="store_true", help="ignore "
                           "the first line in the embeddings file?")
    parsed_args = argparser.parse_args()
    main(parsed_args)
