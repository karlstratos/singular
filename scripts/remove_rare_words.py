# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to remove rare word types in a corpus.
"""
import argparse
from collections import Counter

def build_rare_dictionary(corpus_path, cutoff):
    """Build a dictionary of rare word types."""
    word_count = Counter()
    num_words = 0
    with open(corpus_path, "r") as corpus:
        for line in corpus:
            tokens = line.split()
            for token in tokens:
                word_count[token] += 1
                num_words += 1

    print("{0} words, {1} word types".format(num_words, len(word_count)))

    rare_dictionary = {}
    for word in word_count:
        if word_count[word] <= cutoff:
            rare_dictionary[word] = True

    print("{0} word types considered rare".format(len(rare_dictionary)))
    print("\t => {0} + 1 word types".format(len(word_count) -
                                            len(rare_dictionary)))
    return rare_dictionary

def write_filtered_corpus(corpus_path, rare_dictionary, rare_symbol,
                          output_path):
    """Output a filtered corpus."""
    with open(output_path, "w") as output:
        with open(corpus_path, "r") as corpus:
            for line in corpus:
                tokens = line.split()
                for token in tokens:
                    filtered_token = rare_symbol if token in rare_dictionary \
                                     else token
                    output.write(filtered_token+" ")
                output.write("\n")

def main(args):
    """
    Determine rare word types and replace their occurrences with a special
    symbol.
    """
    rare_dictionary = build_rare_dictionary(args.corpus_path, args.cutoff)

    write_filtered_corpus(args.corpus_path, rare_dictionary, args.rare_symbol,
                          args.output_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("corpus_path", type=str, help="path to corpus")
    argparser.add_argument("cutoff", type=int, help="word types with occurrence"
                           " <= this number are replaced by a special symbol.")
    argparser.add_argument("output_path", type=str, help="path to output")
    argparser.add_argument("--rare_symbol", type=str, default="<?>",
                           help="special symbol for rare word types.")
    parsed_args = argparser.parse_args()
    main(parsed_args)
