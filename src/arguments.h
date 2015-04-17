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

    // Returns the path to a corpus (file or a directory of files).
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

    // Returns the context definition.
    string context_definition() { return context_definition_; }

    // Returns the target dimension of word vectors.
    size_t dim() { return dim_; }

    // Returns the data transformation method.
    string transformation_method() { return transformation_method_; }

    // Returns the scaling method.
    string scaling_method() { return scaling_method_; }

    // Returns the number of context types to hash.
    size_t num_context_hashed() { return num_context_hashed_; }

    // Returns the pseudocount for smoothing.
    size_t pseudocount() { return pseudocount_; }

    // Returns the flag for printing messages to stderr.
    bool verbose() { return verbose_; }

private:
    // Path to a corpus.
    string corpus_path_;

    // Output directory.
    string output_directory_;

    // Recompute counts from scratch.
    bool from_scratch_ = false;

    // Rare word cutoff.
    size_t rare_cutoff_ = 10;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the context window.
    size_t window_size_ = 11;

    // Context definition.
    string context_definition_ = "bag";

    // Target dimension of word vectors.
    size_t dim_ = 500;

    // Data transformation method.
    string transformation_method_ = "sqrt";

    // Scaling method.
    string scaling_method_ = "cca";

    // Number of context types to hash (0 means no hashing).
    size_t num_context_hashed_ = 0;

    // Pseudocount for smoothing.
    size_t pseudocount_ = 0;

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // ARGUMENTS_H_
