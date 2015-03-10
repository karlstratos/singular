// Author: Karl Stratos (stratos@cs.columbia.edu)
//
//  Code for counting word occurrences.

#ifndef COUNTER_H
#define COUNTER_H

#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef size_t Word;
typedef size_t Context;

class Counter {
public:
    // Initializes empty.
    Counter() { }

    // Initializes with an output directory.
    Counter(const string &output_directory) {
	PrepareOutputDirectory(output_directory);
    }

    ~Counter() { }

    // Prepares an output directory.
    void PrepareOutputDirectory(const string &output_directory);

    // Extracts statistics from a corpus.
    void ExtractStatistics(const string &corpus_file);

    // Returns the special string for representing rare words.
    string kRareString() { return kRareString_; }

    // Returns the special string for buffering.
    string kBufferString() { return kBufferString_; }

    // Sets the rare word cutoff value.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets the flag for indicating that there is a sentence per line in the
    // text corpus.
    void set_sentence_per_line(bool sentence_per_line) {
	sentence_per_line_ = sentence_per_line;
    }

    // Sets the context window size.
    void set_window_size(size_t window_size) { window_size_ = window_size; }

    // Sets the context definition.
    void set_context_definition(string context_definition) {
	context_definition_ = context_definition;
    }

private:
    // Extracts the count of each word type appearing in the given corpus file.
    void CountWords(const string &corpus_file);

    // Adds the word to the word dictionary if not already known.
    Word AddWordIfUnknown(const string &word_string);

    // Determines rare word types.
    void DetermineRareWords();

    // Slides a window across a corpus to collect statistics.
    void SlideWindow(const string &corpus_file);

    void FinishWindow(size_t word_index,
		      const vector<string> &position_markers,
		      deque<string> *window,
		      unordered_map<Word, double> *count_word,
		      unordered_map<Context, double> *count_context,
		      unordered_map<Context, unordered_map<Word, double> >
		      *count_word_context);

    // Increments word/context counts from a window of text.
    void ProcessWindow(const deque<string> &window,
		       size_t word_index,
		       const vector<string> &position_markers,
		       unordered_map<Word, double> *count_word,
		       unordered_map<Context, double> *count_context,
		       unordered_map<Context, unordered_map<Word, double> >
		       *count_word_context);

    // Adds the context to the context dictionary if not already known.
    Context AddContextIfUnknown(const string &context_string);

    // Returns a string signature of tunable parameters.
    //    version=0: rare_cutoff_
    //    version=1: 0 + sentence_per_line_, window_size_, context_defintion_
    string Signature(size_t version);

    // Returns the path to the corpus information file.
    string CorpusInfoPath() { return output_directory_ + "/corpus_info"; }

    // Returns the path to the log file.
    string LogPath() { return output_directory_ + "/log." + Signature(1); }

    // Returns the path to the sorted word types file.
    string SortedWordTypesPath() {
	return output_directory_ + "/sorted_word_types";
    }

    // Returns the path to the rare word file.
    string RarePath() {
	return output_directory_ + "/rare_words." + Signature(0);
    }

    // Returns the path to the str2num mapping for words.
    string WordStr2NumPath() {
	return output_directory_ + "/word_str2num." + Signature(0);
    }

    // Returns the path to the str2num mapping for context.
    string ContextStr2NumPath() {
	return output_directory_ + "/context_str2num." + Signature(1);
    }

    // Returns the path to the word-context count file.
    string CountWordContextPath() {
	return output_directory_ + "/count_word_context." + Signature(1);
    }

    // Returns the path to the word count file.
    string CountWordPath() {
	return output_directory_ + "/count_word." + Signature(0);
    }

    // Returns the path to the context count file.
    string CountContextPath() {
	return output_directory_ + "/count_context." + Signature(1);
    }

    // Maps a word string to an integer ID.
    unordered_map<string, Word> word_str2num_;

    // Maps a word integer ID to its original string form.
    unordered_map<Word, string> word_num2str_;

    // Maps a context string to an integer ID.
    unordered_map<string, Context> context_str2num_;

    // Maps a context integer ID to its original string form.
    unordered_map<Context, string> context_num2str_;

    // Path to the log file.
    ofstream log_;

    // Special string for representing rare words.
    const string kRareString_ = "<?>";

    // Special string for representing the out-of-sentence buffer.
    const string kBufferString_ = "<!>";

    // Path to the output directory.
    string output_directory_;

    // If a word type appears <= this number, treat it as a rare symbol.
    size_t rare_cutoff_ = 10;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the context for computing co-occurrences. Note that it needs to
    // be odd if we want the left and right context to have the same length.
    size_t window_size_ = 11;

    // Context definition.
    string context_definition_ = "bag";
};

#endif  // COUNTER_H
