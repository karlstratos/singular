// Author: Karl Stratos (karlstratos@gmail.com)

#include "src/arguments.h"
#include "src/wordrep.h"

int main (int argc, char* argv[]) {
    ArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    // Initialize a WordRep object with an output directory.
    WordRep wordrep(argparser.output_directory());

    // Set variables.
    wordrep.set_rare_cutoff(argparser.rare_cutoff());
    wordrep.set_sentence_per_line(argparser.sentence_per_line());
    wordrep.set_window_size(argparser.window_size());
    wordrep.set_dynamic_context_weight(argparser.dynamic_context_weight());
    wordrep.set_context_definition(argparser.context_definition());
    wordrep.set_dim(argparser.dim());
    wordrep.set_transformation_method(argparser.transformation_method());
    wordrep.set_scaling_method(argparser.scaling_method());
    wordrep.set_smooth_value(argparser.smooth_value());
    wordrep.set_weighting_method(argparser.weighting_method());
    wordrep.set_max_num_epochs(argparser.max_num_epochs());
    wordrep.set_regularization_term(argparser.regularization_term());
    wordrep.set_learning_rate_prior(argparser.learning_rate_prior());

    // If given a corpus, extract statistics from it.
    if (!argparser.corpus_path().empty()) {
	if (argparser.from_scratch()) {  // Recompute counts from scratch.
	    wordrep.ResetOutputDirectory();
	}
	wordrep.ExtractStatistics(argparser.corpus_path());
    }

    // Induce word representations from cached statistics.
    wordrep.InduceLexicalRepresentations();
}
