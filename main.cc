// Author: Karl Stratos (karlstratos@gmail.com)

#include "src/arguments.h"
#include "src/wordrep.h"

int main (int argc, char* argv[]) {
    ArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    if (!argparser.corpus_path().empty() &&
	!argparser.output_directory().empty()) {
	CanonWord canonword;
	canonword.set_output_directory(argparser.output_directory());
	canonword.set_rare_cutoff(argparser.rare_cutoff());
	canonword.set_window_size(argparser.window_size());
	canonword.set_sentence_per_line(argparser.sentence_per_line());
	canonword.set_cca_dim(argparser.cca_dim());
	canonword.set_smoothing_term(argparser.smoothing_term());
	canonword.InduceLexicalRepresentations(argparser.corpus_path());
    }
}
