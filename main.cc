// Author: Karl Stratos (karlstratos@gmail.com)

#include "src/arguments.h"
#include "src/wordrep.h"

int main (int argc, char* argv[]) {
    ArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    CanonWord canonword(argparser.output_directory());
    canonword.set_rare_cutoff(argparser.rare_cutoff());
    canonword.set_window_size(argparser.window_size());
    canonword.set_bag_of_words(argparser.bag_of_words());
    canonword.set_sentence_per_line(argparser.sentence_per_line());
    if (!argparser.corpus_path().empty()) {
	// If given a corpus, extract statistics from it.
	if (argparser.from_scratch()) { canonword.ResetOutputDirectory(); }
	canonword.ExtractStatistics(argparser.corpus_path());
    }
    // Induce word representations from the statistics in the output directory.
    canonword.set_cca_dim(argparser.cca_dim());
    canonword.set_smoothing_term(argparser.smoothing_term());
    canonword.InduceLexicalRepresentations();
}
