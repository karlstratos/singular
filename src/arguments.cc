// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "arguments.h"

void ArgumentProcessor::ParseArguments(int argc, char* argv[]) {
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--corpus") {
	    corpus_path_ = argv[++i];
	} else if (arg == "--output") {
	    output_directory_ = argv[++i];
	} else if (arg == "--force" || arg == "-f") {
	    from_scratch_ = true;
	} else if (arg == "--rare") {
	    rare_cutoff_ = stol(argv[++i]);
	} else if (arg == "--sentences") {
	    sentence_per_line_ = true;
	} else if (arg == "--window") {
	    window_size_ = stol(argv[++i]);
	} else if (arg == "--context") {
	    context_definition_ = argv[++i];
	} else if (arg == "--dim") {
	    dim_ = stol(argv[++i]);
	} else if (arg == "--transform") {
	    transformation_method_ = argv[++i];
	} else if (arg == "--scale") {
	    scaling_method_ = argv[++i];
	} else if (arg == "--hash") {
	    num_context_hashed_ = stol(argv[++i]);
	} else if (arg == "--pseudocount") {
	    pseudocount_ = stol(argv[++i]);
	} else if (arg == "--ce") {
	    context_smoothing_exponent_ = stod(argv[++i]);
	} else if (arg == "--se") {
	    singular_value_exponent_ = stod(argv[++i]);
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose_ = false;
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--corpus [-]:       \t"
	     << "path to a text file or a directory of text files" << endl;

	cout << "--output [-]:       \t"
	     << "path to an output directory" << endl;

	cout << "--force, -f:         \t"
	     << "forcefully recompute from scratch" << endl;

	cout << "--rare [" << rare_cutoff_ << "]:     \t"
	     << "rare word count threshold" << endl;

	cout << "--sentences:        \t"
	     << "have a sentence per line in the corpus" << endl;

	cout << "--window [" << window_size_ << "]:    \t"
	     << "size of the sliding window" << endl;

	cout << "--context [" << context_definition_ << "]: \t"
	     << "context definition: bag, bigram, skipgram, list, baglist"
	     << endl;

	cout << "--dim [" << dim_ << "]:        \t"
	     << "dimensionality of word vectors" << endl;

	cout << "--transform [" << transformation_method_ << "]: \t"
	     << "data transform: raw, sqrt, two-thirds, log"
	     << endl;

	cout << "--scale [" << scaling_method_ << "]:    \t"
	     << "data scaling: raw, cca, reg, ppmi" << endl;

	cout << "--hash [" << num_context_hashed_ << "]:          \t"
	     << "hash size for context (0 means no hashing)" << endl;

	cout << "--pseudocount [" << pseudocount_ << "]:  \t"
	     << "pseudocount for smoothing" << endl;

	cout << "--ce [" << context_smoothing_exponent_ << "]:    \t"
	     << "context smoothing exponent" << endl;

	cout << "--se [" << singular_value_exponent_ << "]:       \t"
	     << "singular value exponent" << endl;

	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr" << endl;

	cout << "--help, -h:           \t"
	     << "show options and quit" << endl;

	exit(0);
    }
}
