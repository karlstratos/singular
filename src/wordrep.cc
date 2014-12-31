// Author: Karl Stratos (karlstratos@gmail.com)

#include "wordrep.h"

#include <deque>
#include <iomanip>
#include <map>

#include "sparsecca.h"

void CanonWord::SetOutputDirectory(const string &output_directory) {
    ASSERT(!output_directory.empty(), "Empty output directory.");
    output_directory_ = output_directory;

    // Prepare the output directory and the log file.
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);
    log_.open(LogPath(), ios::out);
    log_ << fixed << setprecision(2);
}

void CanonWord::ResetOutputDirectory() {
    ASSERT(!output_directory_.empty(), "No output directory given.");
    ASSERT(system(("rm -f " + output_directory_ + "/*").c_str()) == 0,
	   "Cannot remove the content in: " << output_directory_);
    log_.close();
    SetOutputDirectory(output_directory_);
}

void CanonWord::ExtractStatistics(const string &corpus_file) {
    FileManipulator file_manipulator;
    if (!file_manipulator.exists(SortedWordTypesPath())) {
	// Only count word types if there is no previous record.
	CountWords(corpus_file);
    }
    DetermineRareWords();
    ComputeCovariance(corpus_file);
}

void CanonWord::InduceLexicalRepresentations() {
    ASSERT(rare_cutoff_ >= 0, "Shouldn't have negative cutoff at this point");
    FileManipulator file_manipulator;
    ASSERT(file_manipulator.exists(CountWordContextPath()), "File not found, "
	   "read from the corpus: " << CountWordContextPath());
    ASSERT(file_manipulator.exists(CountWordPath()), "File not found, "
	   "read from the corpus: " << CountWordPath());
    ASSERT(file_manipulator.exists(CountContextPath()), "File not found, "
	   "read from the corpus: " << CountContextPath());
    ASSERT(file_manipulator.exists(WordStr2NumPath()), "File not found, "
	   "read from the corpus: " << WordStr2NumPath());
    ASSERT(file_manipulator.exists(SortedWordTypesPath()), "File not found, "
	   "read from the corpus: " << SortedWordTypesPath());

    // Perform CCA on the computed counts.
    time_t begin_time_cca = time(NULL);  // CCA time.
    log_ << "[Performing CCA on the computed counts]" << endl;
    log_ << "   CCA dimension: " << cca_dim_ << endl;

    if (smoothing_term_ < 0) {
	// If smoothing term is negative, set it based on the cutoff value
	// (i.e., the smallest count) we used.
	smoothing_term_ = rare_cutoff_;
	log_ << "   Smoothing term: " << smoothing_term_ << " (automatically "
	     << "set)" << endl << flush;
    } else {
	log_ << "   Smoothing term: " << smoothing_term_ << endl << flush;
    }

    SparseCCASolver sparsecca_solver(cca_dim_, smoothing_term_);
    sparsecca_solver.PerformCCA(CountWordContextPath(),
				CountWordPath(), CountContextPath());
    double time_cca = difftime(time(NULL), begin_time_cca);
    if (sparsecca_solver.rank() < cca_dim_) {
	log_ << "   (*WARNING*) The correlation matrix has rank "
	     << sparsecca_solver.rank() << " < " << cca_dim_ << "!" << endl;
    }
    StringManipulator string_manipulator;
    singular_values_ = *sparsecca_solver.cca_correlations();
    log_ << "   Condition number: "
	 << singular_values_[0] / singular_values_[cca_dim_ - 1] << endl;
    log_ << "   Time taken: " << string_manipulator.print_time(time_cca)
	 << endl;

    string line;
    vector<string> tokens;
    if (word_num2str_.size() == 0 || word_str2num_.size() == 0) {
	// Read the word-integer dictionary if we don't have it.
	word_str2num_.clear();
	ifstream word_str2num_file(WordStr2NumPath(), ios::in);
	while (word_str2num_file.good()) {
	    getline(word_str2num_file, line);
	    if (line == "") { continue; }
	    string_manipulator.split(line, " ", &tokens);
	    word_num2str_[stoi(tokens[1])] = tokens[0];
	    word_str2num_[tokens[0]] = stoi(tokens[1]);
	}
    }
    if (wordcount_.size() == 0) {
	// Read the word counts if we don't have them.
	ifstream sorted_word_types_file(SortedWordTypesPath(), ios::in);
	while (sorted_word_types_file.good()) {
	    getline(sorted_word_types_file, line);
	    if (line == "") { continue; }
	    string_manipulator.split(line, " ", &tokens);
	    string word_string = tokens[0];
	    size_t word_count = stoi(tokens[1]);
	    if (word_str2num_.find(word_string) != word_str2num_.end()) {
		wordcount_[word_string] = word_count;
	    } else {
		wordcount_[kRareString_] += word_count;
	    }
	}
    }
    // Sort word types in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount(wordcount_.begin(),
						   wordcount_.end());
    sort(sorted_wordcount.begin(), sorted_wordcount.end(),
	 sort_pairs_second<string, size_t, greater<size_t> >());

    // Collect and normalize the singular vectors for word representations.
    log_ << endl << "[Collecting word vectors from singular vectors]" << endl;
    log_ << "   Normalizing each word vector..." << endl << flush;
    Eigen::MatrixXd *word_matrix = sparsecca_solver.cca_transformation_x();
    for (size_t i = 0; i < word_matrix->cols(); ++i) {  // column = word
	word_matrix->col(i).normalize();  // Normalize each column.
    }

    // Put the vectors in the PCA basis.
    log_ << "   Putting word vectors in the PCA basis..." << endl << flush;
    for (size_t i = 0; i < word_matrix->rows(); ++i) {  // Centering.
	double row_mean = word_matrix->row(i).mean();
	for (size_t j = 0; j < word_matrix->cols(); ++j) {
	    (*word_matrix)(i, j) -= row_mean;
	}
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> eigen_svd(*word_matrix,
						Eigen::ComputeThinV);
    Eigen::VectorXd pca_singular_values = eigen_svd.singularValues();
    Eigen::VectorXd variance = pca_singular_values;
    ofstream pca_variance_file(PCAVariancePath(), ios::out);
    for (size_t i = 0; i < variance.size(); ++i) {
	variance(i) *= variance(i);
	variance(i) /= word_matrix->cols() - 1;
	pca_variance_file << variance(i) << endl;
    }

    Eigen::MatrixXd word_matrix_pca = eigen_svd.matrixV();
    word_matrix_pca.transposeInPlace();
    for (size_t i = 0; i < word_matrix_pca.rows(); ++i) {
	word_matrix_pca.row(i) *= pca_singular_values(i);
    }

    wordvectors_.clear();
    for (Word word = 0; word < word_num2str_.size(); ++word) {
	string word_string = word_num2str_[word];
	wordvectors_[word_string] = word_matrix_pca.col(word);
    }

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

    // Sort word types in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount(wordcount_.begin(),
						   wordcount_.end());
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
    corpus_info_file << num_words_ << " words" << endl;
    corpus_info_file << sorted_wordcount.size() << " word types" << endl;
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
    // Read in the sorted word type counts.
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    vector<pair<string, size_t> > sorted_wordcount;
    ifstream sorted_word_types_file(SortedWordTypesPath(), ios::in);
    wordcount_.clear();
    num_words_ = 0;
    while (sorted_word_types_file.good()) {
	getline(sorted_word_types_file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	string word_string = tokens[0];
	size_t word_count = stoi(tokens[1]);
	sorted_wordcount.push_back(make_pair(word_string, word_count));
	wordcount_[word_string] += word_count;
	num_words_ += word_count;
    }
    log_ << "[Original corpus]" << endl;
    log_ << "   " << num_words_ << " words" << endl;
    log_ << "   " << sorted_wordcount.size() << " word types" << endl;

    unordered_map<string, bool> rare;  // Keys correspond to rare word types.
    if (rare_cutoff_ >= 0) {
	// If a non-negative cutoff value is specified, use it.
	for (int i = sorted_wordcount.size() - 1; i >= 0; --i) {
	    string word_string = sorted_wordcount[i].first;
	    size_t word_count = sorted_wordcount[i].second;
	    if (word_count > rare_cutoff_) { break; }
	    rare[word_string] = true;
	}
    } else {
	// If no cutoff value is specified, let the model decide some cutoff
	// value (> 0) for determining rare words.
	size_t accumulated_count = 0;
	vector<string> currently_considered_words;
	size_t current_count =
	    sorted_wordcount[sorted_wordcount.size() - 1].second;
	for (int i = sorted_wordcount.size() - 1; i >= 0; --i) {
	    string word_string = sorted_wordcount[i].first;
	    size_t word_count = sorted_wordcount[i].second;
	    if (word_count == current_count) {
		// Collect all words with the same count.
		currently_considered_words.push_back(word_string);
		accumulated_count += word_count;
	    } else {
		// Upon a different word count, add word types collected so far
		// as rare words, and decide whether to continue thresholding.
		rare_cutoff_ = current_count;
		for (string considered_word: currently_considered_words) {
		    rare[considered_word] = true;
		}
		currently_considered_words.clear();
		double rare_mass =
		    ((double) accumulated_count * 100) / num_words_;
		//cout << rare_cutoff_ << endl << rare_mass << endl;
		if (rare_mass > 2.0) {
		    // If the rare word mass is more than 5% of the unigram
		    // mass, stop thresholding.
		    break;
		}
		current_count = word_count;
	    }
	}
    }

    double total_count_nonrare = num_words_;
    if (rare.size() > 0) {
	// If we have rare words, record them and remap the word integer IDs.
	ofstream rare_file(RarePath(), ios::out);
	size_t total_count_rare = 0;
	for (const auto &string_pair: rare) {
	    string rare_string = string_pair.first;
	    size_t rare_count = wordcount_[rare_string];
	    rare_file << rare_string << " " << rare_count << endl;
	    total_count_rare += rare_count;
	}
	total_count_nonrare -= total_count_rare;

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

    log_ << endl << "[Processed with cutoff " << rare_cutoff_ << "]" << endl;
    log_ << "   " << rare.size() << " rare word types grouped to "
	 << kRareString_ << endl;
    log_ << "   Now " << wordcount_.size() << " word types" << endl;
    log_ << "   Preserved " << total_count_nonrare / num_words_ * 100
	 << "% of the unigram mass" << endl;

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
    log_ << "   " << num_nonzeros << " nonzeros" << endl << endl <<  flush;

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
    for (Word word = 0; word < count_word.size(); ++word) {
	count_word_file << count_word[word] << endl;
    }

    ofstream count_context_file(CountContextPath(), ios::out);
    for (Context context = 0; context < count_context.size(); ++context) {
	count_context_file << count_context[context] << endl;
    }
}
