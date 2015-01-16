# Author: Karl Stratos (karlstratos@gmail.com)
"""
This module is used to display similar words (in consine similarity).

Argument: [word embeddings file]
"""
import sys
from numpy import array
from numpy import dot
from numpy import linalg

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

def show_similar(embedding_path):
    """Interactively displays similar words (in consine similarity)."""
    embedding, dim = read_normalized_embeddings(embedding_path)
    print "Read {0} embeddings of dimension {1}".format(len(embedding), dim)

    while True:
        try:
            word = raw_input("Type a word (or just quit the program): ")
            if not word in embedding:
                print "There is no embedding for word \"{0}\"".format(word)
            else:
                neighbors = []
                for other_word in embedding:
                    if other_word == word:
                        continue
                    cosine = dot(embedding[word], embedding[other_word])
                    neighbors.append((cosine, other_word))
                neighbors.sort(reverse=True)
                for i in range(30):
                    cosine, buddy = neighbors[i]
                    print '\t\t{0:.4f}\t\t{1}'.format(cosine, buddy)
        except (KeyboardInterrupt, EOFError):
            print
            exit(0)

if __name__ == "__main__":
    # Path to word embeddings file.
    EMBEDDING_PATH = sys.argv[1]

    show_similar(EMBEDDING_PATH)
