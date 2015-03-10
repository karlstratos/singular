// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "counter.h"

#include <iomanip>
#include <map>

#include "sparsesvd.h"
#include "util.h"

void Counter::PrepareOutputDirectory(const string &output_directory) {
    ASSERT(!output_directory.empty(), "Empty output directory.");
    output_directory_ = output_directory;

    // Prepare the output directory and the log file.
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);

    log_.open(LogPath(), ios::out);
    log_ << fixed << setprecision(3);
}

void Counter::ExtractStatistics(const string &corpus_file) {
    CountWords(corpus_file);
    DetermineRareWords();
    SlideWindow(corpus_file);
}

void Counter::CountWords(const string &corpus_file) {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.Exists(SortedWordTypesPath())) { return; }

    ASSERT(window_size_ >= 2, "Window size less than 2: " << window_size_);
    ifstream file(corpus_file, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << corpus_file);
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    unordered_map<string, size_t> wordcount;
    size_t num_words = 0;
    while (file.good()) {
	getline(file, line);
	if (line == "") { continue; }
	string_manipulator.Split(line, " ", &tokens);
	for (const string &token : tokens) {
	    ASSERT(token != kRareString_, "Rare symbol present: " << token);
	    ASSERT(token != kBufferString_, "Buffer symbol present: " << token);
	    AddWordIfUnknown(token);
	    ++wordcount[token];
	    ++num_words;
	}
    }
    ASSERT(num_words >= window_size_, "Number of words in the corpus smaller "
	   "than the window size: " << num_words << " < " << window_size_);

    // Sort word types in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount(wordcount.begin(),
						   wordcount.end());
    sort(sorted_wordcount.begin(), sorted_wordcount.end(),
	 sort_pairs_second<string, size_t, greater<size_t> >());

    ofstream sorted_word_types_file(SortedWordTypesPath(), ios::out);
    for (size_t i = 0; i < sorted_wordcount.size(); ++i) {
	string word_string = sorted_wordcount[i].first;
	size_t word_frequency = sorted_wordcount[i].second;
	sorted_word_types_file << word_string << " " << word_frequency << endl;
    }

    // Write a corpus information file.
    ofstream corpus_info_file(CorpusInfoPath(), ios::out);
    corpus_info_file << "Path: " << corpus_file << endl;
    corpus_info_file << num_words << " words" << endl;
    corpus_info_file << sorted_wordcount.size() << " word types" << endl;
}

Word Counter::AddWordIfUnknown(const string &word_string) {
    ASSERT(!word_string.empty(), "Adding an empty string for word!");
    if (word_str2num_.find(word_string) == word_str2num_.end()) {
	Word word = word_str2num_.size();
	word_str2num_[word_string] = word;
	word_num2str_[word] = word_string;
    }
    return word_str2num_[word_string];
}

void Counter::DetermineRareWords() {
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;

    // Read in the original word dictionary and sorted word type counts.
    word_str2num_.clear();
    word_num2str_.clear();
    size_t num_words = 0;
    vector<pair<string, size_t> > sorted_wordcount;
    ifstream sorted_word_types_file(SortedWordTypesPath(), ios::in);
    while (sorted_word_types_file.good()) {
	getline(sorted_word_types_file, line);
	if (line == "") { continue; }
	string_manipulator.Split(line, " ", &tokens);
	string word_string = tokens[0];
	size_t word_count = stol(tokens[1]);
	sorted_wordcount.push_back(make_pair(word_string, word_count));
	AddWordIfUnknown(word_string);
	num_words += word_count;
    }
    log_ << "[Corpus]" << endl;
    log_ << "   Number of words: " << num_words << endl;
    log_ << "   Cutoff: " << rare_cutoff_ << ", " << word_str2num_.size()
	 << " => ";

    // Filter the dictionary by grouping rare types into a single type.
    unordered_map<string, bool> rare;
    size_t num_rare = 0;
    ofstream rare_file(RarePath(), ios::out);
    word_str2num_.clear();
    word_num2str_.clear();
    for (int i = sorted_wordcount.size() - 1; i >= 0; --i) {
	string word_string = sorted_wordcount[i].first;
	size_t word_count = sorted_wordcount[i].second;
	if (word_count <= rare_cutoff_) {
	    rare[word_string] = true;
	    num_rare += word_count;
	    rare_file << word_string << " " << word_count << endl;
	    AddWordIfUnknown(kRareString_);
	} else {
	    AddWordIfUnknown(word_string);
	}
    }
    double preserved_unigram_mass =
	((double) num_words - num_rare) / num_words * 100;

    log_ << word_str2num_.size() << " word types" << endl;
    log_ << "   Uncut: " << preserved_unigram_mass
	 << "% unigram mass" << endl;

    // Write the filtered word dictionary.
    ofstream word_str2num_file(WordStr2NumPath(), ios::out);
    for (const auto &word_pair: word_str2num_) {
	word_str2num_file << word_pair.first << " " << word_pair.second << endl;
    }
}

void Counter::SlideWindow(const string &corpus_file) {
    string corpus_format = (sentence_per_line_) ? "1 line = 1 sentence" :
	"Whole Text = 1 sentence";
    log_ << endl << "[Sliding window]" << endl;
    log_ << "   Window size: " << window_size_ << endl;
    log_ << "   Context definition: " << context_definition_ << endl;
    log_ << "   Corpus format: " << corpus_format << endl << flush;

    // If we already have count files, do not repeat the work.
    FileManipulator file_manipulator;
    if (file_manipulator.Exists(ContextStr2NumPath()) &&
	file_manipulator.Exists(CountWordContextPath()) &&
	file_manipulator.Exists(CountWordPath()) &&
	file_manipulator.Exists(CountContextPath())) {
	log_ << "   Counts already exist" << endl;
	return;
    }

    // Pre-compute values we need over and over again.
    size_t word_index = (window_size_ - 1) / 2;  // Right-biased
    vector<string> position_markers(window_size_);
    for (size_t context_index = 0; context_index < window_size_;
	 ++context_index) {
	if (context_index != word_index) {
	    position_markers[context_index] = "w(" +
		to_string(((int) context_index) - ((int) word_index)) + ")=";
	}
    }

    // count_word_context[j][i] = count of word i and context j coocurring
    unordered_map<Context, unordered_map<Word, double> > count_word_context;
    unordered_map<Word, double> count_word;  // i-th: count of word i
    unordered_map<Context, double> count_context;  // j-th: count of context j

    // Put start buffering in the window.
    deque<string> window;
    for (size_t buffering = 0; buffering < word_index; ++buffering) {
	window.push_back(kBufferString_);
    }

    time_t begin_time_sliding = time(NULL);  // Window sliding time.
    ifstream file(corpus_file, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << corpus_file);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    while (file.good()) {
	getline(file, line);
	if (line == "") { continue; }
	string_manipulator.Split(line, " ", &tokens);
	for (const string &token : tokens) {
	    // TODO: Switch to checking rare dictionary?
	    string new_string =
		(word_str2num_.find(token) != word_str2num_.end()) ?
		token : kRareString_;
	    window.push_back(new_string);
	    if (window.size() >= window_size_) {  // Full window.
		ProcessWindow(window, word_index, position_markers,
			      &count_word, &count_context, &count_word_context);
		window.pop_front();
	    }
	}

	if (sentence_per_line_) {
	    FinishWindow(word_index, position_markers, &window,
			 &count_word, &count_context, &count_word_context);
	}
    }

    if (!sentence_per_line_) {
	FinishWindow(word_index, position_markers, &window,
		     &count_word, &count_context, &count_word_context);
    }

    double time_sliding = difftime(time(NULL), begin_time_sliding);
    log_ << "   Time taken: " << string_manipulator.TimeString(time_sliding)
	 << endl;

    // Write the filtered context dictionary.
    ofstream context_str2num_file(ContextStr2NumPath(), ios::out);
    for (const auto &context_pair: context_str2num_) {
	context_str2num_file << context_pair.first << " "
			     << context_pair.second << endl;
    }

    // Write counts to the output directory.
    SparseSVDSolver sparsesvd_solver;  // Write as a sparse matrix for SVDLIBC.
    sparsesvd_solver.WriteSparseMatrix(count_word_context,
				       CountWordContextPath());
    ofstream count_word_file(CountWordPath(), ios::out);
    for (Word word = 0; word < count_word.size(); ++word) {
	count_word_file << count_word[word] << endl;
    }
    ofstream count_context_file(CountContextPath(), ios::out);
    for (Context context = 0; context < count_context.size(); ++context) {
	count_context_file << count_context[context] << endl;
    }
}

void Counter::FinishWindow(size_t word_index,
			   const vector<string> &position_markers,
			   deque<string> *window,
			   unordered_map<Word, double> *count_word,
			   unordered_map<Context, double> *count_context,
			   unordered_map<Context, unordered_map<Word, double> >
			   *count_word_context) {
    size_t original_window_size = window->size();
    while (window->size() < window_size_) {
	// First fill up the window in case the sentence was short.
	(*window).push_back(kBufferString_);  //   [<!> a] -> [<!> a <!> ]
    }
    for (size_t buffering = word_index; buffering < original_window_size;
	 ++buffering) {
	ProcessWindow(*window, word_index, position_markers,
		      count_word, count_context, count_word_context);
	(*window).pop_front();
	(*window).push_back(kBufferString_);
    }
    (*window).clear();
    for (size_t buffering = 0; buffering < word_index; ++buffering) {
	(*window).push_back(kBufferString_);
    }
}

void Counter::ProcessWindow(const deque<string> &window,
			    size_t word_index,
			    const vector<string> &position_markers,
			    unordered_map<Word, double> *count_word,
			    unordered_map<Context, double> *count_context,
			    unordered_map<Context, unordered_map<Word, double> >
			    *count_word_context) {
    Word word = word_str2num_[window.at(word_index)];
    (*count_word)[word] += 1;

    for (size_t context_index = 0; context_index < window.size();
	 ++context_index) {
	if (context_index == word_index) { continue; }
	string context_string = window.at(context_index);
	if (context_definition_ == "bag") {  // Bag-of-words (BOW).
	    Context bag_context = AddContextIfUnknown(context_string);
	    (*count_context)[bag_context] += 1;
	    (*count_word_context)[bag_context][word] += 1;
	} else if (context_definition_ == "list") {  // List-of-words (LOW)
	    Context list_context =
		AddContextIfUnknown(position_markers.at(context_index) +
				    context_string);
	    (*count_context)[list_context] += 1;
	    (*count_word_context)[list_context][word] += 1;
	} else if (context_definition_ == "baglist") {  // BOW+LOW
	    Context bag_context = AddContextIfUnknown(context_string);
	    Context list_context =
		AddContextIfUnknown(position_markers.at(context_index) +
				    context_string);
	    (*count_context)[bag_context] += 1;
	    (*count_context)[list_context] += 1;
	    (*count_word_context)[bag_context][word] += 1;
	    (*count_word_context)[list_context][word] += 1;
	} else {
	    ASSERT(false, "Unknown context definition: " <<
		   context_definition_);
	}
    }
}

Context Counter::AddContextIfUnknown(const string &context_string) {
    ASSERT(!context_string.empty(), "Adding an empty string for context!");
    if (context_str2num_.find(context_string) == context_str2num_.end()) {
	Context context = context_str2num_.size();
	context_str2num_[context_string] = context;
	context_num2str_[context] = context_string;
    }
    return context_str2num_[context_string];
}

string Counter::Signature(size_t version) {
    ASSERT(version <= 2, "Unrecognized signature version: " << version);

    string signature = "rare" + to_string(rare_cutoff_);
    if (version >= 1) {
	if (sentence_per_line_) {
	    signature += "_spl";
	}
	signature += "_window" + to_string(window_size_);
	signature += "_" + context_definition_;
    }
    return signature;
}
