// Author: Karl Stratos (stratos@cs.columbia.edu)
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

    // Returns the size of the context window.
    size_t window_size() { return window_size_; }

    // Returns the flag for weighting context dynamically.
    bool dynamic_context_weight() { return dynamic_context_weight_; }

    // Returns the context definition.
    string context_definition() { return context_definition_; }

    // Returns the target dimension of word vectors.
    size_t dim() { return dim_; }

    // Returns the flag for smoothing context counts.
    bool context_smoothing() { return context_smoothing_; }

    // Returns the data transformation method.
    string transformation_method() { return transformation_method_; }

    // Returns the scaling method.
    string scaling_method() { return scaling_method_; }

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

    // Size of the context window.
    size_t window_size_ = 3;

    // Weight context dynamically?
    bool dynamic_context_weight_ = false;

    // Context definition.
    string context_definition_ = "bag";

    // Target dimension of word vectors.
    size_t dim_ = 100;

    // Smooth context counts?
    bool context_smoothing_ = false;

    // Data transformation method.
    string transformation_method_ = "raw";

    // Scaling method.
    string scaling_method_ = "cca";
};

#endif  // ARGUMENTS_H_
