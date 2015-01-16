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
	    rare_cutoff_ = stoi(argv[++i]);
	} else if (arg == "--window") {
	    window_size_ = stoi(argv[++i]);
	} else if (arg == "--bag-of-words") {
	    bag_of_words_ = true;
	} else if (arg == "--sentence-per-line") {
	    sentence_per_line_ = true;
	} else if (arg == "--cca-dim") {
	    cca_dim_ = stoi(argv[++i]);
	} else if (arg == "--smoothing") {
	    smoothing_term_ = stoi(argv[++i]);
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
	     << "Rare word cutoff value (-1 lets the model decide)."
	     << endl << endl;

	cout << "--window "
	     << "(default: " << window_size_ << ")" << endl
	     << "Size of the context to compute covariance on." << endl << endl;

	cout << "--bag-of-words" << endl
	     << "Use bag-of-words (i.e., not position sensitive) context?"
	     << endl << endl;

	cout << "--sentence-per-line " << endl
	     << "Have a sentence per line in the text corpus?" << endl << endl;

	cout << "--cca-dim "
	     << "(default: " << cca_dim_ << ")" << endl
	     << "Dimension of the CCA subspace." << endl << endl;

	cout << "--smoothing "
	     << "(default: " << smoothing_term_ << ")" << endl
	     << "Smoothing term for calculating the correlation matrix "
	     << "(-1 lets the model decide)."  << endl << endl;

	exit(0);
    }
}
