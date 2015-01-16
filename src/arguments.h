// Author: Karl Stratos (karlstratos@gmail.com)
//
// Code for processing parser arguments.

#ifndef ARGUMENTS_H_
#define ARGUMENTS_H_

#include <iostream>
#include <string>

using namespace std;

// Class for processing user-specified arguments.
class ArgumentProcessor {
public:
    // Parse the command line strings into arguments.
    void ParseArguments(int argc, char* argv[]);

    // Returns the path to a text corpus.
    string corpus_path() { return corpus_path_; }

    // Returns the output directory.
    string output_directory() { return output_directory_; }

    // Returns the flag for recomputing all counts from scratch.
    bool from_scratch() { return from_scratch_; }

    // Returns the rare word cutoff value (-1 lets the model decide).
    int rare_cutoff() { return rare_cutoff_; }

    // Returns the size of the context to compute covariance on.
    size_t window_size() { return window_size_; }

    // Returns the flag for using bag-of-words context.
    bool bag_of_words() { return bag_of_words_; }

    // Returns the flag for indicating that there is a sentence per line in the
    // text corpus.
    bool sentence_per_line() { return sentence_per_line_; }

    // Returns the dimension of the CCA subspace.
    size_t cca_dim() { return cca_dim_; }

    // Returns the smoothing term for calculating the correlation matrix (-1
    // lets the model decide).
    int smoothing_term() { return smoothing_term_; }

private:
    // Path to a text corpus.
    string corpus_path_;

    // Output directory.
    string output_directory_;

    // Recompute counts from scratch (no caching)?
    bool from_scratch_ = false;

    // Rare word cutoff.
    int rare_cutoff_ = -1;  // Let the model decide.

    // Size of the context to compute covariance on.
    size_t window_size_ = 3;  // 1 word to the left and to the right.

    // Use bag-of-words (i.e., not position sensitive) context?
    bool bag_of_words_ = false;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Dimension of the CCA subspace.
    size_t cca_dim_ = 1000;

    // Smoothing term for calculating the correlation matrix.
    int smoothing_term_ = -1;  // Let the model decide.
};

#endif  // ARGUMENTS_H_
