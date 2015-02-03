// Author: Karl Stratos (karlstratos@gmail.com)
//
// Class for processing user-specified arguments.

#ifndef ARGUMENTS_H_
#define ARGUMENTS_H_

#include <iostream>
#include <string>

using namespace std;

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

    // Returns the rare word cutoff value.
    size_t rare_cutoff() { return rare_cutoff_; }

    // Returns the size of the context to compute covariance on.
    size_t window_size() { return window_size_; }

    // Returns the flag for using bag-of-words context.
    bool bag_of_words() { return bag_of_words_; }

    // Returns the flag for indicating that there is a sentence per line in the
    // text corpus.
    bool sentence_per_line() { return sentence_per_line_; }

    // Returns the target dimension of word vectors.
    size_t dim() { return dim_; }

    // Returns the scaling method.
    string scaling_method() { return scaling_method_; }

    // Returns the smoothing value.
    size_t smooth_value() { return smooth_value_; }

    // Returns the weighting method.
    string weighting_method() { return weighting_method_; }

private:
    // Path to a text corpus.
    string corpus_path_;

    // Output directory.
    string output_directory_;

    // Recompute counts from scratch.
    bool from_scratch_ = false;

    // Rare word cutoff.
    size_t rare_cutoff_ = 1;

    // Size of the context to compute covariance on.
    size_t window_size_ = 3;

    // Use bag-of-words context?
    bool bag_of_words_ = false;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Target dimension of word vectors.
    size_t dim_ = 100;

    // Scaling method.
    string scaling_method_ = "cca";

    // Smoothing value.
    size_t smooth_value_ = 5;

    // Weighting method.
    string weighting_method_;
};

#endif  // ARGUMENTS_H_
