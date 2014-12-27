// Author: Karl Stratos (karlstratos@gmail.com)

#include "arguments.h"

void ArgumentProcessor::ParseArguments(int argc, char* argv[]) {
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--corpus_path") {
	    corpus_path_ = argv[++i];
	} else if (arg == "--output_directory") {
	    output_directory_ = argv[++i];
	} else if (arg == "--rare_cutoff") {
	    rare_cutoff_ = stoi(argv[++i]);
	} else if (arg == "--window_size") {
	    window_size_ = stoi(argv[++i]);
	} else if (arg == "--sentence_per_line") {
	    sentence_per_line_ = true;
	} else if (arg == "--cca_dim") {
	    cca_dim_ = stoi(argv[++i]);
	} else if (arg == "--smoothing_term") {
	    smoothing_term_ = stod(argv[++i]);
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1)
    {
	cout << "ARGUMENTS:" << endl << endl;

	cout << "--corpus_path [file]" << endl
	     << "Path to a text corpus." << endl << endl;

	cout << "--output_directory [directory]" << endl
	     << "Path to the output directory." << endl << endl;

	cout << "--rare_cutoff "
	     << "(default: " << rare_cutoff_ << ")" << endl
	     << "Rare word cutoff value (-1 lets the model decide)."
	     << endl << endl;

	cout << "--window_size "
	     << "(default: " << window_size_ << ")" << endl
	     << "Size of the context to compute covariance on." << endl << endl;

	cout << "--sentence_per_line "
	     << "Have a sentence per line in the text corpus?" << endl << endl;

	cout << "--cca_dim "
	     << "(default: " << cca_dim_ << ")" << endl
	     << "Dimension of the CCA subspace." << endl << endl;

	cout << "--smoothing_term "
	     << "(default: " << smoothing_term_ << ")" << endl
	     << "Smoothing term for calculating the correlation matrix "
	     << "(-1 lets the model decide)."  << endl << endl;

	exit(0);
    }
}
