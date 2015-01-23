// Author: Karl Stratos (karlstratos@gmail.com)

#include "src/arguments.h"
#include "src/wordrep.h"

int main (int argc, char* argv[]) {
    ArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    WordRep wordrep(argparser.output_directory());
    wordrep.set_rare_cutoff(argparser.rare_cutoff());
    wordrep.set_window_size(argparser.window_size());
    wordrep.set_bag_of_words(argparser.bag_of_words());
    wordrep.set_sentence_per_line(argparser.sentence_per_line());
    if (!argparser.corpus_path().empty()) {
	// If given a corpus, extract statistics from it.
	if (argparser.from_scratch()) { wordrep.ResetOutputDirectory(); }
	wordrep.ExtractStatistics(argparser.corpus_path());
    }
    // Induce word representations from cached statistics.
    wordrep.set_dim(argparser.dim());
    wordrep.set_smooth_value(argparser.smooth_value());
    wordrep.set_scaling_method(argparser.scaling_method());
    wordrep.InduceLexicalRepresentations();
}
