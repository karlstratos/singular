// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "src/arguments.h"
#include "src/counter.h"

int main (int argc, char* argv[]) {
    CountArgumentProcessor argparser;
    argparser.ParseArguments(argc, argv);

    Counter counter(argparser.output_directory());
    counter.set_rare_cutoff(argparser.rare_cutoff());
    counter.set_sentence_per_line(argparser.sentence_per_line());
    counter.set_window_size(argparser.window_size());
    counter.set_context_definition(argparser.context_definition());
    counter.ExtractStatistics(argparser.corpus_path());
}
