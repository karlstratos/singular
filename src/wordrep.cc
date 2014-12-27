// Author: Karl Stratos (karlstratos@gmail.com)

#include "wordrep.h"

#include <deque>
#include <iomanip>
#include <map>

#include "sparsecca.h"

void CanonWord::InduceLexicalRepresentations(const string &corpus_file) {
    // Prepare the output directory and the log file.
    ASSERT(!output_directory_.empty(), "Empty output directory.");
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);
    ASSERT(system(("rm -f " + output_directory_ + "/*").c_str()) == 0,
	   "Cannot remove the content in: " << output_directory_);
    log_.open(LogPath(), ios::out);
    log_ << fixed << setprecision(2);

    StringManipulator string_manipulator;
    time_t begin_time_total = time(NULL);  // Total time.

    // Compute word counts from the corpus with appropriate preprocessing.
    log_ << "Corpus: " << corpus_file << endl << flush;
    time_t begin_time_count = time(NULL);  // Counting time.
    CountWords(corpus_file);
    DetermineRareWords();
    ComputeCovariance(corpus_file);
    double time_count = difftime(time(NULL), begin_time_count);

    // Perform CCA on the computed counts.
    log_ << endl << "[Performing CCA on the computed counts]" << endl;
    log_ << "   CCA dimension: " << cca_dim_ << endl;
    log_ << "   Smoothing term: " << smoothing_term_ << endl << flush;
    time_t begin_time_cca = time(NULL);  // CCA time.
    SparseCCASolver sparsecca_solver(cca_dim_, smoothing_term_);
    sparsecca_solver.PerformCCA(CountWordContextPath(),
				CountWordPath(), CountContextPath());

    // Collect and normalize the singular vectors for word representations.
    wordvectors_.clear();
    for (Word word = 0; word < word_num2str_.size(); ++word) {
	string word_string = word_num2str_[word];
	// Each column corresponds to a word type.
	wordvectors_[word_string] =
	    sparsecca_solver.cca_transformation_x()->col(word);
	wordvectors_[word_string].normalize();  // Normalize length.
    }
    singular_values_ = *sparsecca_solver.cca_correlations();
    double time_cca = difftime(time(NULL), begin_time_cca);
    log_ << "   Condition number: " << singular_values_[0] /
	singular_values_[cca_dim_ - 1] << endl;

    log_ << endl << "Writing results in: " << output_directory_ << endl
	 << flush;
    // Sort word types in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount(wordcount_.begin(),
						   wordcount_.end());
    sort(sorted_wordcount.begin(), sorted_wordcount.end(),
	 sort_pairs_second<string, size_t, greater<size_t> >());

    // Write word vectors.
    ofstream wordvectors_file(WordVectorsPath(), ios::out);
    for (size_t i = 0; i < wordcount_.size(); ++i) {
	string word_string = sorted_wordcount[i].first;
	size_t word_frequency = sorted_wordcount[i].second;
	wordvectors_file << word_frequency << " " << word_string;
	for (size_t j = 0; j < wordvectors_[word_string].size(); ++ j) {
	    wordvectors_file << " " << wordvectors_[word_string](j);
	}
	wordvectors_file << endl;
    }

    // Write singular values.
    ofstream singular_values_file(SingularValuesPath(), ios::out);
    for (size_t i = 0; i < singular_values_.size(); ++i) {
	singular_values_file << singular_values_[i] << endl;
    }

    double time_total = difftime(time(NULL), begin_time_total);
    log_ << "Time spent in counting: "
	 << string_manipulator.print_time(time_count) << endl;
    log_ << "Time spent in CCA and normalizing: "
	 << string_manipulator.print_time(time_cca) << endl;
    log_ << "Total time: " << string_manipulator.print_time(time_total)
	 << endl;
}

Word CanonWord::word_str2num(const string &word_string) {
    ASSERT(word_str2num_.find(word_string) != word_str2num_.end(),
	   "Requesting integer ID of an unknown word string: " << word_string);
    return word_str2num_[word_string];
}

string CanonWord::word_num2str(Word word) {
    ASSERT(word_num2str_.find(word) != word_num2str_.end(),
	   "Requesting string of an unknown word integer: " << word);
    return word_num2str_[word];
}

Context CanonWord::context_str2num(const string &context_string) {
    ASSERT(context_str2num_.find(context_string) != context_str2num_.end(),
	   "Requesting integer ID of an unknown context string: "
	   << context_string);
    return context_str2num_[context_string];
}

string CanonWord::context_num2str(Context context) {
    ASSERT(context_num2str_.find(context) != context_num2str_.end(),
	   "Requesting string of an unknown context integer: " << context);
    return context_num2str_[context];
}

void CanonWord::CountWords(const string &corpus_file) {
    ASSERT(window_size_ >= 2, "Window size less than 2: " << window_size_);
    wordcount_.clear();
    ifstream file(corpus_file, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << corpus_file);
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    num_words_ = 0;
    while (file.good()) {
	getline(file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	for (const string &token : tokens) {
	    ASSERT(token != kRareString_, "Rare symbol present: " << token);
	    ASSERT(token != kBufferString_, "Buffer symbol present: " << token);
	    AddWordIfUnknown(token);
	    ++wordcount_[token];
	    ++num_words_;
	}
    }
    ASSERT(num_words_ >= window_size_, "Number of words in the corpus smaller "
	   "than the window size: " << num_words_ << " < " << window_size_);
    log_ << endl << "[Original corpus]" << endl;
    log_ << "   " << num_words_ << " words with " << wordcount_.size()
	 << " distinct word types" << endl << flush;
}

Word CanonWord::AddWordIfUnknown(const string &word_string) {
    ASSERT(!word_string.empty(), "Adding an empty string for word!");
    if (word_str2num_.find(word_string) == word_str2num_.end()) {
	Word word = word_str2num_.size();
	word_str2num_[word_string] = word;
	word_num2str_[word] = word_string;
    }
    return word_str2num_[word_string];
}

Context CanonWord::AddContextIfUnknown(const string &context_string) {
    ASSERT(!context_string.empty(), "Adding an empty string for context!");
    if (context_str2num_.find(context_string) == context_str2num_.end()) {
	Context context = context_str2num_.size();
	context_str2num_[context_string] = context;
	context_num2str_[context] = context_string;
    }
    return context_str2num_[context_string];
}

void CanonWord::DetermineRareWords() {
    // Sort word types in increasing frequency.
    vector<pair<string, size_t> > sorted_wordcount(wordcount_.begin(),
						   wordcount_.end());
    sort(sorted_wordcount.begin(), sorted_wordcount.end(),
	 sort_pairs_second<string, size_t>());

    // Keys of this dictionary correspond to rare word types.
    unordered_map<string, bool> rare;
    size_t cutoff_used = 0;
    if (rare_cutoff_ >= 0) {
	// If a non-negative cutoff value is specified, use it.
	cutoff_used = rare_cutoff_;
	for (size_t i = 0; i < wordcount_.size(); ++i) {
	    string word_string = sorted_wordcount[i].first;
	    size_t word_frequency = sorted_wordcount[i].second;
	    if (word_frequency > rare_cutoff_) { break; }
	    rare[word_string] = true;
	}
    } else {
	// If no cutoff value is specified, let the model decide rare words.
	size_t accumulated_count = 0;
	vector<string> currently_considered_words;
	size_t current_frequency = sorted_wordcount[0].second;
	for (size_t i = 0; i < wordcount_.size(); ++i) {
	    string word_string = sorted_wordcount[i].first;
	    size_t word_frequency = sorted_wordcount[i].second;
	    currently_considered_words.push_back(word_string);
	    accumulated_count += word_frequency;
	    if (word_frequency > current_frequency) {
		// Upon a different word count, add infrequent word types
		// collected so far as rare words, and decide whether to
		// continuestop thresholding.
		double rare_mass =
		    ((double) accumulated_count * 100) / num_words_;
		cutoff_used = current_frequency;
		for (string considered_word: currently_considered_words) {
		    rare[considered_word] = true;
		}
		currently_considered_words.clear();
		if (rare_mass > 0.8) {
		    // If the rare word mass is more than 0.8% of the unigram
		    // mass, stop thresholding.
		    break;
		}
	    }
	}
    }

    if (rare.size() > 0) {
	// If we have rare words, record them and remap the word integer IDs.
	ofstream rare_file(RarePath(), ios::out);
	for (const auto &string_pair: rare) {
	    string rare_string = string_pair.first;
	    size_t rare_count = wordcount_[rare_string];
	    rare_file << rare_string << " " << rare_count << endl;
	}

	unordered_map<string, size_t> wordcount_copy = wordcount_;
	wordcount_.clear();
	word_str2num_.clear();
	word_num2str_.clear();
	AddWordIfUnknown(kRareString_);  // Add a special symbol for rare words.
	for (const auto &word_string_pair: wordcount_copy) {
	    string word_string = word_string_pair.first;
	    size_t word_count = word_string_pair.second;
	    if (rare.find(word_string) != rare.end()) {
		word_string = kRareString_;
	    }
	    AddWordIfUnknown(word_string);
	    wordcount_[word_string] += word_count;
	}
    }

    log_ << endl << "[Processed with cutoff " << cutoff_used << "]" << endl;
    log_ << "   " << rare.size() << " rare word types grouped to "
	 << kRareString_ << endl;
    log_ << "   Now " << wordcount_.size() << " distinct word types" << endl;

    if (smoothing_term_ < 0.0) {
	// If smoothing term is negative, set it based on the cutoff value
	// (i.e., the smallest count) we used.
	smoothing_term_ = cutoff_used;
	log_ << "   Also automatically set the smoothing term: "
	     << smoothing_term_ << endl;
    }

    // Write the word-integer mapping.
    ofstream word_str2num_file(WordStr2NumPath(), ios::out);
    for (const auto &word_pair: word_str2num_) {
	word_str2num_file << word_pair.first << " " << word_pair.second << endl;
    }
}

void CanonWord::ComputeCovariance(const string &corpus_file) {
    // Figure out the indices of the current and context words.
    size_t word_index = (window_size_ % 2 == 0) ?
	(window_size_ / 2) - 1 : window_size_ / 2;  // Right-biased
    vector<size_t> context_indices;
    vector<string> position_markers(window_size_);
    for (size_t context_index = 0; context_index < window_size_;
	 ++context_index) {
	if (context_index != word_index) {
	    context_indices.push_back(context_index);
	    position_markers[context_index] = "w(" +
		to_string(((int) context_index) - ((int) word_index)) + ")=";
	}
    }

    // Put start buffering in the window.
    deque<string> window;
    for (size_t buffering = 0; buffering < word_index; ++buffering) {
	window.push_back(kBufferString_);
    }

    // count_word_context[j][i] = count of word i and context j coocurring
    unordered_map<Word, unordered_map<Word, double> > count_word_context;
    unordered_map<Word, double> count_word;  // i-th: count of word i
    unordered_map<Word, double> count_context;  // j-th: count of context j
    size_t num_nonzeros = 0;  // Number of nonzero values in count_word_context.

    log_ << endl << "[Counting...";
    if (sentence_per_line_) {
	log_ << " (1 line = 1 sentence)";
    } else {
	log_ << " (entire text = 1 sentence)";
    }
    log_ << "]" << endl << flush;
    ifstream file(corpus_file, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << corpus_file);
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    while (file.good()) {
	getline(file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	for (const string &token : tokens) {
	    string new_string =
		(word_str2num_.find(token) != word_str2num_.end()) ?
		token : kRareString_;
	    window.push_back(new_string);
	    if (window.size() >= window_size_) {
		 // Collect statistics from the full window.
		Word word = word_str2num_[window[word_index]];
		++count_word[word];
		for (Word context_index : context_indices) {
		    string context_string = position_markers[context_index] +
			window[context_index];
		    Context context = AddContextIfUnknown(context_string);
		    ++count_context[context];
		    if (count_word_context[context][word] == 0) {
			++num_nonzeros;  // New nonzero value.
		    }
		    ++count_word_context[context][word];
		}
		window.pop_front();
	    }
	}

	if (sentence_per_line_) {
	    // Put end buffering and collect counts.
	    for (size_t buffering = word_index + 1; buffering < window_size_;
		 ++buffering) {
		window.push_back(kBufferString_);
		Word word = word_str2num_[window[word_index]];
		++count_word[word];
		for (Word context_index : context_indices) {
		    string context_string = position_markers[context_index] +
			window[context_index];
		    Context context = AddContextIfUnknown(context_string);
		    ++count_context[context];
		    if (count_word_context[context][word] == 0) {
			++num_nonzeros;  // New nonzero value.
		    }
		    ++count_word_context[context][word];
		}
		window.pop_front();
	    }
	    window.clear();
	    for (size_t buffering = 0; buffering < word_index; ++buffering) {
		window.push_back(kBufferString_);
	    }
	}
    }

    if (!sentence_per_line_) {
	// Put end buffering and collect counts.
	for (size_t buffering = word_index + 1; buffering < window_size_;
	     ++buffering) {
	    window.push_back(kBufferString_);
	    Word word = word_str2num_[window[word_index]];
	    ++count_word[word];
	    for (Word context_index : context_indices) {
		string context_string = position_markers[context_index] +
		    window[context_index];
		Context context = AddContextIfUnknown(context_string);
		++count_context[context];
		if (count_word_context[context][word] == 0) {
		    ++num_nonzeros;  // New nonzero value.
		}
		++count_word_context[context][word];
	    }
	    window.pop_front();
	}
    }
    log_ << "   " << count_word.size() << " by " << count_context.size()
	 << endl;
    log_ << "   " << num_nonzeros << " nonzero values" << endl << flush;

    // Write the context-integer mapping.
    ofstream context_str2num_file(ContextStr2NumPath(), ios::out);
    for (const auto &context_pair: context_str2num_) {
	context_str2num_file << context_pair.first << " "
			     << context_pair.second << endl;
    }

    // Write the covariance values to an output directory.
    SparseSVDSolver sparsesvd_solver;  // Write a sparse matrix for SVDLIBC.
    sparsesvd_solver.WriteSparseMatrix(count_word_context,
				       CountWordContextPath());

    ofstream count_word_file(CountWordPath(), ios::out);
    for (const auto &word_pair: count_word) {
	ASSERT(word_num2str_[word_pair.first] != kBufferString_,
	       "Buffer string cannot occur as a word: something is wrong.");
	count_word_file << word_pair.first << " " << word_pair.second
			<< endl;
    }

    ofstream count_context_file(CountContextPath(), ios::out);
    for (const auto &context_pair: count_context) {
	count_context_file << context_pair.first << " " << context_pair.second
			   << endl;
    }
}
