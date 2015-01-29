// Author: Karl Stratos (karlstratos@gmail.com)

#include "arguments.h"

void ArgumentProcessor::ParseArguments(int argc, char* argv[]) {
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--corpus") {
	    corpus_path_ = argv[++i];
	} else if (arg == "--output") {
	    output_directory_ = argv[++i];
	} else if (arg == "--recompute") {
	    from_scratch_ = true;
	} else if (arg == "--rare") {
	    rare_cutoff_ = stol(argv[++i]);
	} else if (arg == "--window") {
	    window_size_ = stol(argv[++i]);
	} else if (arg == "--bag-of-words") {
	    bag_of_words_ = true;
	} else if (arg == "--sentence-per-line") {
	    sentence_per_line_ = true;
	} else if (arg == "--dim") {
	    dim_ = stol(argv[++i]);
	} else if (arg == "--scaling") {
	    scaling_method_ = argv[++i];
	} else if (arg == "--smooth") {
	    smooth_value_ = stol(argv[++i]);
	} else if (arg == "-h" || arg == "--help"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1)
    {
	cout << "ARGUMENTS:" << endl << endl;

	cout << "--corpus [file]" << endl
	     << "Path to a text corpus." << endl << endl;

	cout << "--output [directory]" << endl
	     << "Path to the output directory." << endl << endl;

	cout << "--recompute" << endl
	     << "Recompute all counts from scratch?" << endl << endl;

	cout << "--rare "
	     << "(default: " << rare_cutoff_ << ")" << endl
	     << "Rare word cutoff value." << endl << endl;

	cout << "--window "
	     << "(default: " << window_size_ << ")" << endl
	     << "Size of the context to compute covariance on." << endl << endl;

	cout << "--bag-of-words" << endl
	     << "Use bag-of-words context?"
	     << endl << endl;

	cout << "--sentence-per-line " << endl
	     << "Have a sentence per line in the text corpus?" << endl << endl;

	cout << "--dim "
	     << "(default: " << dim_ << ")" << endl
	     << "Target dimension of word vectors." << endl << endl;

	cout << "--scaling "
	     << "(default: " << scaling_method_ << ")" << endl
	     << "Scaling method: cca, ppmi." << endl << endl;

	cout << "--smooth "
	     << "(default: " << smooth_value_ << ")" << endl
	     << "Smoothing value." << endl << endl;

	exit(0);
    }
}
