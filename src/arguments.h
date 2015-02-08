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

    // Returns the flag for indicating that there is a sentence per line in the
    // text corpus.
    bool sentence_per_line() { return sentence_per_line_; }

    // Returns the size of the context to compute covariance on.
    size_t window_size() { return window_size_; }

    // Returns the context definition.
    string context_definition() { return context_definition_; }

    // Returns the target dimension of word vectors.
    size_t dim() { return dim_; }

    // Returns the scaling method.
    string scaling_method() { return scaling_method_; }

    // Returns the smoothing value.
    size_t smooth_value() { return smooth_value_; }

    // Returns the weighting method.
    string weighting_method() { return weighting_method_; }

    // Returns the maximum number of training epochs.
    size_t max_num_epochs() { return max_num_epochs_; }

    // Returns the regularization term.
    double regularization_term() { return regularization_term_; }

    // Returns the learning rate prior.
    double learning_rate_prior() { return learning_rate_prior_; }

private:
    // Path to a text corpus.
    string corpus_path_;

    // Output directory.
    string output_directory_;

    // Recompute counts from scratch.
    bool from_scratch_ = false;

    // Rare word cutoff.
    size_t rare_cutoff_ = 1;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the context to compute covariance on.
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "bag";

    // Target dimension of word vectors.
    size_t dim_ = 100;

    // Scaling method.
    string scaling_method_ = "cca";

    // Smoothing value.
    size_t smooth_value_ = 5;

    // Weighting method.
    string weighting_method_ = "";

    // Maximum number of training epochs.
    size_t max_num_epochs_ = 100;

    // Regularization term.
    double regularization_term_ = 0.1;

    // Learning rate prior.
    double learning_rate_prior_ = 0.1;
};

#endif  // ARGUMENTS_H_
