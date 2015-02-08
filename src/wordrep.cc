// Author: Karl Stratos (karlstratos@gmail.com)

#include "wordrep.h"

#include <deque>
#include <dirent.h>
#include <iomanip>
#include <map>

#include "cluster.h"

void WordRep::SetOutputDirectory(const string &output_directory) {
    ASSERT(!output_directory.empty(), "Empty output directory.");
    output_directory_ = output_directory;

    // Prepare the output directory and the log file.
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);

    // Obtain the latest version marker for log.
    size_t latest_version = 0;
    DIR *directory = opendir(output_directory_.c_str());
    struct dirent *entry;
    while (NULL != (entry = readdir(directory))) {
	string file_name = entry->d_name;
	if (file_name.substr(0, 3) == "log") {
	    size_t version = stol(file_name.substr(4));
	    if (version > latest_version) { latest_version = version; }
	}
    }
    closedir(directory);
    log_.open(LogPath() + "." + to_string(latest_version + 1), ios::out);
    log_ << fixed << setprecision(3);
}

void WordRep::ResetOutputDirectory() {
    ASSERT(!output_directory_.empty(), "No output directory given.");
    ASSERT(system(("rm -f " + output_directory_ + "/*").c_str()) == 0,
	   "Cannot remove the content in: " << output_directory_);
    log_.close();
    SetOutputDirectory(output_directory_);
}

void WordRep::ExtractStatistics(const string &corpus_file) {
    CountWords(corpus_file);
    DetermineRareWords();
    SlideWindow(corpus_file);
}

void WordRep::InduceLexicalRepresentations() {
    // Load a filtered word dictionary from a cached file.
    LoadWordDictionary();

    // Load a sorted list of word-count pairs from a cached file.
    LoadSortedWordCounts();

    // Induce word vectors from cached count files.
    InduceWordVectors();

    // Test the quality of word vectors on simple tasks.
    TestQualityOfWordVectors();

    // Perform greedy agglomerative clustering over word vectors.
    PerformAgglomerativeClustering(dim_);

    // Rotate word vectors to PCA coordinates.
    RotateWordVectorsToPCACoordinates();
}

Word WordRep::word_str2num(const string &word_string) {
    ASSERT(word_str2num_.find(word_string) != word_str2num_.end(),
	   "Requesting integer ID of an unknown word string: " << word_string);
    return word_str2num_[word_string];
}

string WordRep::word_num2str(Word word) {
    ASSERT(word_num2str_.find(word) != word_num2str_.end(),
	   "Requesting string of an unknown word integer: " << word);
    return word_num2str_[word];
}

Context WordRep::context_str2num(const string &context_string) {
    ASSERT(context_str2num_.find(context_string) != context_str2num_.end(),
	   "Requesting integer ID of an unknown context string: "
	   << context_string);
    return context_str2num_[context_string];
}

string WordRep::context_num2str(Context context) {
    ASSERT(context_num2str_.find(context) != context_num2str_.end(),
	   "Requesting string of an unknown context integer: " << context);
    return context_num2str_[context];
}

void WordRep::CountWords(const string &corpus_file) {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.exists(SortedWordTypesPath())) { return; }

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
	string_manipulator.split(line, " ", &tokens);
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

Word WordRep::AddWordIfUnknown(const string &word_string) {
    ASSERT(!word_string.empty(), "Adding an empty string for word!");
    if (word_str2num_.find(word_string) == word_str2num_.end()) {
	Word word = word_str2num_.size();
	word_str2num_[word_string] = word;
	word_num2str_[word] = word_string;
    }
    return word_str2num_[word_string];
}

Context WordRep::AddContextIfUnknown(const string &context_string) {
    ASSERT(!context_string.empty(), "Adding an empty string for context!");
    if (context_str2num_.find(context_string) == context_str2num_.end()) {
	Context context = context_str2num_.size();
	context_str2num_[context_string] = context;
	context_num2str_[context] = context_string;
    }
    return context_str2num_[context_string];
}

void WordRep::DetermineRareWords() {
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
	string_manipulator.split(line, " ", &tokens);
	string word_string = tokens[0];
	size_t word_count = stol(tokens[1]);
	sorted_wordcount.push_back(make_pair(word_string, word_count));
	AddWordIfUnknown(word_string);
	num_words += word_count;
    }
    log_ << "[Corpus]" << endl;
    log_ << "   " << num_words << " words" << endl;
    log_ << "   Cutoff " << rare_cutoff_ << ": " << word_str2num_.size()
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

void WordRep::SlideWindow(const string &corpus_file) {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.exists(ContextStr2NumPath()) &&
	file_manipulator.exists(CountWordContextPath()) &&
	file_manipulator.exists(CountWordPath()) &&
	file_manipulator.exists(CountContextPath())) { return; }
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    string context_definition = (bag_of_words_) ? "bag-of-words" :
	"position-sensitive";
    string corpus_format = (sentence_per_line_) ? "1 line = 1 sentence" :
	"Whole Text = 1 sentence";
    log_ << endl << "[Sliding window]" << endl;
    log_ << "   Window size: " << window_size_ << endl;
    log_ << "   Context: " << context_definition << endl;
    log_ << "   Corpus format: " <<  corpus_format << endl << flush;

    // Figure out the indices of the current and context words.
    size_t word_index = (window_size_ - 1) / 2;  // Right-biased
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

    // count_word_context[j][i] = count of word i and context j coocurring
    unordered_map<Word, unordered_map<Word, double> > count_word_context;
    unordered_map<Word, double> count_word;  // i-th: count of word i
    unordered_map<Word, double> count_context;  // j-th: count of context j

    // Put start buffering in the window.
    deque<string> window;
    for (size_t buffering = 0; buffering < word_index; ++buffering) {
	window.push_back(kBufferString_);
    }

    time_t begin_time_sliding = time(NULL);  // Window sliding time.
    ifstream file(corpus_file, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << corpus_file);
    while (file.good()) {
	getline(file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	for (const string &token : tokens) {
	    // TODO: Switch to checking rare dictionary?
	    string new_string =
		(word_str2num_.find(token) != word_str2num_.end()) ?
		token : kRareString_;
	    window.push_back(new_string);
	    if (window.size() >= window_size_) {
		// Collect statistics from the full window.
		Word word = word_str2num_[window[word_index]];
		++count_word[word];
		for (Word context_index : context_indices) {
		    string context_string = window[context_index];
		    if (!bag_of_words_) {
			context_string =
			    position_markers[context_index] + context_string;
		    }
		    Context context = AddContextIfUnknown(context_string);
		    ++count_context[context];
		    ++count_word_context[context][word];
		}
		window.pop_front();
	    }
	}

	if (sentence_per_line_) {  // End-buffer and collect counts.
	    while (window.size() < window_size_) {
		// But first fill up the window.
		window.push_back(kBufferString_);
	    }
	    for (size_t buffering = word_index + 1; buffering < window_size_;
		 ++buffering) {
		window.push_back(kBufferString_);
		Word word = word_str2num_[window[word_index]];
		++count_word[word];
		for (Word context_index : context_indices) {
		    string context_string = window[context_index];
		    if (!bag_of_words_) {
			context_string =
			    position_markers[context_index] + context_string;
		    }
		    Context context = AddContextIfUnknown(context_string);
		    ++count_context[context];
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

    if (!sentence_per_line_) {  // End-buffer and collect counts.
	while (window.size() < window_size_) {
	    // But first fill up the window.
	    window.push_back(kBufferString_);
	}
	for (size_t buffering = word_index + 1; buffering < window_size_;
	     ++buffering) {
	    window.push_back(kBufferString_);
	    Word word = word_str2num_[window[word_index]];
	    ++count_word[word];
	    for (Word context_index : context_indices) {
		string context_string = window[context_index];
		if (!bag_of_words_) {
		    context_string =
			position_markers[context_index] + context_string;
		}
		Context context = AddContextIfUnknown(context_string);
		++count_context[context];
		++count_word_context[context][word];
	    }
	    window.pop_front();
	}
    }
    double time_sliding = difftime(time(NULL), begin_time_sliding);
    log_ << "   Time taken: " << string_manipulator.print_time(time_sliding)
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

void WordRep::InduceWordVectors() {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (!file_manipulator.exists(WordVectorsPath())) {
	Eigen::MatrixXd word_matrix = CalculateWordMatrix();
	ASSERT(word_matrix.cols() == sorted_wordcount_.size(), "Word matrix "
	       "dimension and vocabulary size mismatch: " << word_matrix.cols()
	       << " vs " << sorted_wordcount_.size());

	ofstream wordvectors_file(WordVectorsPath(), ios::out);
	for (size_t i = 0; i < sorted_wordcount_.size(); ++i) {
	    string word_string = sorted_wordcount_[i].first;
	    size_t word_count = sorted_wordcount_[i].second;
	    Word word = word_str2num_[word_string];
	    word_matrix.col(word).normalize();  // Normalize columns (words).
	    wordvectors_[word_string] = word_matrix.col(word);
	    wordvectors_file << word_count << " " << word_string;
	    for (size_t j = 0; j < wordvectors_[word_string].size(); ++ j) {
		wordvectors_file << " " << wordvectors_[word_string](j);
	    }
	    wordvectors_file << endl;
	}
    } else {  // Load word vectors.
	StringManipulator string_manipulator;
	string line;
	vector<string> tokens;
	wordvectors_.clear();
	ifstream wordvectors_file(WordVectorsPath(), ios::in);
	while (wordvectors_file.good()) {
	    getline(wordvectors_file, line);
	    if (line == "") { continue; }

	    // line = [count] [word_string] [value_{1}] ... [value_{dim_}]
	    string_manipulator.split(line, " ", &tokens);
	    Eigen::VectorXd vector(tokens.size() - 2);
	    for (size_t i = 0; i < tokens.size() - 2; ++i) {
		vector(i) = stod(tokens[i + 2]);
	    }
	    wordvectors_[tokens[1]] = vector;
	}
    }
}

void WordRep::LoadWordDictionary() {
    FileManipulator file_manipulator;
    ASSERT(file_manipulator.exists(WordStr2NumPath()), "File not found, "
	   "read from the corpus: " << WordStr2NumPath());

    word_str2num_.clear();
    word_num2str_.clear();
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    ifstream word_str2num_file(WordStr2NumPath(), ios::in);
    while (word_str2num_file.good()) {
	getline(word_str2num_file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	word_num2str_[stol(tokens[1])] = tokens[0];
	word_str2num_[tokens[0]] = stol(tokens[1]);
    }
}

void WordRep::LoadSortedWordCounts() {
    FileManipulator file_manipulator;
    ASSERT(file_manipulator.exists(SortedWordTypesPath()), "File not found, "
	   "read from the corpus: " << SortedWordTypesPath());
    ASSERT(word_str2num_.size() > 0, "Word dictionary not loaded.");

    ifstream sorted_word_types_file(SortedWordTypesPath(), ios::in);
    string line;
    vector<string> tokens;
    StringManipulator string_manipulator;
    unordered_map<string, size_t> wordcount;
    while (sorted_word_types_file.good()) {
	getline(sorted_word_types_file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	string word_string = tokens[0];
	size_t word_count = stol(tokens[1]);
	if (word_str2num_.find(word_string) != word_str2num_.end()) {
	    wordcount[word_string] = word_count;
	} else {
	    wordcount[kRareString_] += word_count;
	}
    }
    sorted_wordcount_.clear();
    for (const auto &word_count_pair : wordcount) {
	sorted_wordcount_.push_back(word_count_pair);
    }
    sort(sorted_wordcount_.begin(), sorted_wordcount_.end(),
	 sort_pairs_second<string, size_t, greater<size_t> >());
}

Eigen::MatrixXd WordRep::CalculateWordMatrix() {
    FileManipulator file_manipulator;
    ASSERT(file_manipulator.exists(CountWordContextPath()), "File not found, "
	   "read from the corpus: " << CountWordContextPath());
    ASSERT(file_manipulator.exists(CountWordPath()), "File not found, "
	   "read from the corpus: " << CountWordPath());
    ASSERT(file_manipulator.exists(CountContextPath()), "File not found, "
	   "read from the corpus: " << CountContextPath());
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;

    // Get information about the count matrix.
    ifstream count_word_context_file(CountWordContextPath(), ios::in);
    getline(count_word_context_file, line);
    string_manipulator.split(line, " ", &tokens);
    size_t dim1 = stol(tokens[0]);
    size_t dim2 = stol(tokens[1]);
    size_t num_nonzeros = stol(tokens[2]);

    // Get the number of samples (= number of words).
    size_t num_samples = 0;
    ifstream count_word_file(CountWordPath(), ios::in);
    while (count_word_file.good()) {
	getline(count_word_file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	num_samples += stol(tokens[0]);
    }

    log_ << endl << "[Decomposing a matrix of scaled counts]" << endl;
    log_ << "   Matrix: " << dim1 << " x " << dim2 << " (" << num_nonzeros
	 << " nonzeros)" << endl;
    log_ << "   Rank of SVD: " << dim_ << endl;
    log_ << "   Scaling method: " << scaling_method_ << endl;
    if (scaling_method_ == "cca" || scaling_method_ == "scca" ||
	scaling_method_ == "lcca" || scaling_method_ == "rreg") {
	log_ << "   Smoothing value: " << smooth_value_ << endl;
    }
    SMat weights = nullptr;  // Optional weighting.
    if (!weighting_method_.empty()) {
	log_ << "   Weighting method: " << weighting_method_ << endl;
	log_ << "      - Regularization term: " << regularization_term_ << endl;
	log_ << "      - Learning rate prior: " << learning_rate_prior_ << endl;
	log_ << flush;
	weights = GetWeights(weighting_method_);
    }

    time_t begin_time_decomposition = time(NULL);
    Decomposer decomposer(dim_);
    decomposer.set_left_singular_vectors_path(LeftSingularVectorsPath());
    decomposer.set_right_singular_vectors_path(RightSingularVectorsPath());
    decomposer.set_singular_values_path(SingularValuesPath());
    decomposer.set_num_samples(num_samples);
    decomposer.set_scaling_method(scaling_method_);
    decomposer.set_smooth_value(smooth_value_);
    decomposer.set_weights(weights);
    decomposer.set_regularization_term(regularization_term_);
    decomposer.set_learning_rate_prior(learning_rate_prior_);
    decomposer.Decompose(CountWordContextPath(), CountWordPath(),
			 CountContextPath());
    double time_decomposition = difftime(time(NULL), begin_time_decomposition);
    if (decomposer.rank() < dim_) {
	log_ << "   ***WARNING*** The matrix has rank "
	     << decomposer.rank() << " < " << dim_ << "!" << endl;
    }
    svdFreeSMat(weights); // Free the weight matrix.

    singular_values_ = *decomposer.singular_values();
    log_ << "   Condition number: "
	 << singular_values_[0] / singular_values_[dim_ - 1] << endl;
    log_ << "   Time taken: "
	 << string_manipulator.print_time(time_decomposition) << endl;

    return *decomposer.left_matrix();
}

SMat WordRep::GetWeights(const string &weight_method) {
    FileManipulator file_manipulator;
    ASSERT(file_manipulator.exists(CountWordContextPath()),
	   "Can't find word-context count matrix: " << CountWordContextPath());

    // Use SparseSVDSolver to load the word-context count matrix.
    SparseSVDSolver sparsesvd_solver;
    SMat weights =
	sparsesvd_solver.ReadSparseMatrixFromFile(CountWordContextPath());

    double max_value = 0;
    if (weight_method == "anscombe") {
	// Apply the Anscombe transform on counts and determine the max value.
	for (size_t col = 0; col < weights->cols; ++col) {
	    size_t current_column_nonzero_index = weights->pointr[col];
	    size_t next_column_start_nonzero_index = weights->pointr[col + 1];
	    while (current_column_nonzero_index <
		   next_column_start_nonzero_index) {
		weights->value[current_column_nonzero_index] = 2.0 *
		    sqrt(weights->value[current_column_nonzero_index] + 0.375);
		if (weights->value[current_column_nonzero_index] > max_value) {
		    max_value = weights->value[current_column_nonzero_index];
		}
		++current_column_nonzero_index;
	    }
	}
	ASSERT(max_value > 0, "Max count is zero!");
    }


    for (size_t col = 0; col < weights->cols; ++col) {
	size_t current_column_nonzero_index = weights->pointr[col];
	size_t next_column_start_nonzero_index = weights->pointr[col + 1];
	while (current_column_nonzero_index < next_column_start_nonzero_index) {
	    if (weight_method == "anscombe") {
		// Divide by the max value to put values in the range (0, 1].
		weights->value[current_column_nonzero_index] /= max_value;
	    } else if (weight_method == "glove") {
		// Recipe provided in the GloVe paper.
		double x_max = 100.0;
		double alpha = 0.75;
		double x = weights->value[current_column_nonzero_index];
		double scaled_value = (x < x_max) ? pow(x / x_max, alpha) : 1.0;
		weights->value[current_column_nonzero_index] = scaled_value;
	    } else {
		ASSERT(false, "Unknown weighting: " << weighting_method_);
	    }
	    ++current_column_nonzero_index;
	}
    }
    return weights;
}

void WordRep::TestQualityOfWordVectors() {
    string wordsim353_path = "third_party/public_datasets/wordsim353.txt";
    string men_path = "third_party/public_datasets/men.txt";
    string syn_path = "third_party/public_datasets/syntactic_analogies.txt";
    string mixed_path = "third_party/public_datasets/mixed_analogies.txt";
    FileManipulator file_manipulator;
    if (!file_manipulator.exists(wordsim353_path) ||
	!file_manipulator.exists(men_path) ||
	!file_manipulator.exists(syn_path) ||
	!file_manipulator.exists(mixed_path)) {
	// Skip evaluation (e.g., in unit tests) if files are not found.
	return;
    }
    log_ << endl << "[Quality of word vectors]" << endl;

    // Use 3 decimal places for word similartiy.
    log_ << fixed << setprecision(3);

    // Word similarity with wordsim353.txt.
    size_t num_instances_wordsim353;
    size_t num_handled_wordsim353;
    double corr_wordsim353;
    EvaluateWordSimilarity(wordsim353_path, &num_instances_wordsim353,
			   &num_handled_wordsim353, &corr_wordsim353);
    log_ << "   wordsim353: " << corr_wordsim353 << " ("
	 << num_handled_wordsim353 << "/" << num_instances_wordsim353
	 << " evaluated)" << endl;

    // Word similarity with men.txt.
    size_t num_instances_men;
    size_t num_handled_men;
    double corr_men;
    EvaluateWordSimilarity(men_path, &num_instances_men,
			   &num_handled_men, &corr_men);
    log_ << "   MEN:        " << corr_men << " (" << num_handled_men << "/"
	 << num_instances_men << " evaluated)" << endl;
    log_ << fixed << setprecision(2);

    // Word analogy with syntactic_analogies.txt.
    size_t num_instances_syn;
    size_t num_handled_syn;
    double acc_syn;
    EvaluateWordAnalogy(syn_path, &num_instances_syn, &num_handled_syn,
			&acc_syn);
    log_ << "   Syntactic analogies: " << acc_syn << "% (" << num_handled_syn
	 << "/" << num_instances_syn << " evaluated)" << endl;

    // Word analogy with mixed_analogies.txt.
    size_t num_instances_mixed;
    size_t num_handled_mixed;
    double acc_mixed;
    EvaluateWordAnalogy(mixed_path, &num_instances_mixed, &num_handled_mixed,
			&acc_mixed);
    log_ << "   Mixed analogies:     " << acc_mixed << "% ("
	 << num_handled_mixed << "/" << num_instances_mixed << " evaluated)"
	 << endl;
}

void WordRep::EvaluateWordSimilarity(const string &file_path,
				     size_t *num_instances, size_t *num_handled,
				     double *correlation) {
    ifstream similarity_file(file_path, ios::in);
    ASSERT(similarity_file.is_open(), "Cannot open file: " << file_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    vector<double> human_scores;
    vector<double> cosine_scores;
    *num_instances = 0;
    *num_handled = 0;
    while (similarity_file.good()) {
	getline(similarity_file, line);
	if (line == "") { continue; }
	++(*num_instances);
	string_manipulator.split(line, " ", &tokens);
	ASSERT(tokens.size() == 3, "Wrong format for word similarity!");
	string word1 = tokens[0];
	string word2 = tokens[1];
	string word1_lowercase = string_manipulator.lowercase(word1);
	string word2_lowercase = string_manipulator.lowercase(word2);
	double human_score = stod(tokens[2]);

	// Get a vector for each word type. First, try to get a vector for the
	// original string. If not found, try lowercasing.
	Eigen::VectorXd vector_word1, vector_word2;
	if (wordvectors_.find(word1) != wordvectors_.end()) {
	    vector_word1 = wordvectors_[word1];
	} else if (wordvectors_.find(word1_lowercase) != wordvectors_.end()) {
	    vector_word1 = wordvectors_[word1_lowercase];
	}
	if (wordvectors_.find(word2) != wordvectors_.end()) {
	    vector_word2 = wordvectors_[word2];
	} else if (wordvectors_.find(word2_lowercase) != wordvectors_.end()) {
	    vector_word2 = wordvectors_[word2_lowercase];
	}

	// If we have vectors for both word types, compute similarity.
	if (vector_word1.size() > 0 && vector_word2.size() > 0) {
	    // Assumes that word vectors already have length 1.
	    double cosine_score = vector_word1.dot(vector_word2);
	    human_scores.push_back(human_score);
	    cosine_scores.push_back(cosine_score);
	    ++(*num_handled);
	}
    }
    Stat stat;
    *correlation = stat.ComputeSpearman(human_scores, cosine_scores);
}

void WordRep::EvaluateWordAnalogy(const string &file_path,
				  size_t *num_instances, size_t *num_handled,
				  double *accuracy) {
    // Read analogy questions and their vocabulary.
    ifstream analogy_file(file_path, ios::in);
    ASSERT(analogy_file.is_open(), "Cannot open file: " << file_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    vector<tuple<string, string, string, string> > analogies;
    unordered_map<string, Eigen::VectorXd> wordvectors_subset;
    while (analogy_file.good()) {
	getline(analogy_file, line);
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	ASSERT(tokens.size() == 5, "Wrong format for word analogy!");
	// Ignore the analogy category: only compute the overall accuracy.
	string w1 = tokens[1];
	string w1_lowercase = string_manipulator.lowercase(w1);
	string w2 = tokens[2];
	string w2_lowercase = string_manipulator.lowercase(w2);
	string v1 = tokens[3];
	string v1_lowercase = string_manipulator.lowercase(v1);
	string v2 = tokens[4];
	string v2_lowercase = string_manipulator.lowercase(v2);
	analogies.push_back(make_tuple(w1, w2, v1, v2));

	// Get a vector for each word type. First, try to get a vector for the
	// original string. If not found, try lowercasing.
	if (wordvectors_.find(w1) != wordvectors_.end()) {
	    wordvectors_subset[w1] = wordvectors_[w1];
	} else if (wordvectors_.find(w1_lowercase) != wordvectors_.end()) {
	    wordvectors_subset[w1] = wordvectors_[w1_lowercase];
	}
	if (wordvectors_.find(w2) != wordvectors_.end()) {
	    wordvectors_subset[w2] = wordvectors_[w2];
	} else if (wordvectors_.find(w2_lowercase) != wordvectors_.end()) {
	    wordvectors_subset[w2] = wordvectors_[w2_lowercase];
	}
	if (wordvectors_.find(v1) != wordvectors_.end()) {
	    wordvectors_subset[v1] = wordvectors_[v1];
	} else if (wordvectors_.find(v1_lowercase) != wordvectors_.end()) {
	    wordvectors_subset[v1] = wordvectors_[v1_lowercase];
	}
	if (wordvectors_.find(v2) != wordvectors_.end()) {
	    wordvectors_subset[v2] = wordvectors_[v2];
	} else if (wordvectors_.find(v2_lowercase) != wordvectors_.end()) {
	    wordvectors_subset[v2] = wordvectors_[v2_lowercase];
	}
    }

    // For each analogy question "w1:w2 as in v1:v2" such that we have vector
    // representations for word types w1, w2, v1, v2, predict v2.
    *num_instances = 0;
    *num_handled = 0;
    size_t num_correct = 0;
    for (const auto &word_quadruple : analogies) {
	++(*num_instances);
	string w1 = get<0>(word_quadruple);
	string w2 = get<1>(word_quadruple);
	string v1 = get<2>(word_quadruple);
	string v2 = get<3>(word_quadruple);
	if (wordvectors_subset.find(w1) != wordvectors_subset.end() &&
	    wordvectors_subset.find(w2) != wordvectors_subset.end() &&
	    wordvectors_subset.find(v1) != wordvectors_subset.end() &&
	    wordvectors_subset.find(v2) != wordvectors_subset.end()) {
	    ++(*num_handled);
	    string predicted_v2 = AnswerAnalogyQuestion(w1, w2, v1,
							wordvectors_subset);
	    if (predicted_v2 == v2) { ++num_correct; }
	}
    }
    *accuracy = ((double) num_correct) / (*num_handled) * 100.0;
}

string WordRep::AnswerAnalogyQuestion(
    string w1, string w2, string v1,
    const unordered_map<string, Eigen::VectorXd> &wordvectors_subset) {
    ASSERT(wordvectors_subset.find(w1) != wordvectors_subset.end(),
	   "No vector for " << w1);
    ASSERT(wordvectors_subset.find(w2) != wordvectors_subset.end(),
	   "No vector for " << w2);
    ASSERT(wordvectors_subset.find(v1) != wordvectors_subset.end(),
	   "No vector for " << v1);
    // Assumes vectors are already normalized to have length 1.
    Eigen::VectorXd w1_embedding = wordvectors_subset.at(w1);
    Eigen::VectorXd w2_embedding = wordvectors_subset.at(w2);
    Eigen::VectorXd v1_embedding = wordvectors_subset.at(v1);
    string predicted_v2 = "";
    double max_score = -numeric_limits<double>::max();
    for (const auto &word_vector_pair : wordvectors_subset) {
	string word = word_vector_pair.first;
	if (word == w1 || word == w2 || word == v1) { continue; }
	Eigen::VectorXd word_embedding = word_vector_pair.second;
	double shifted_cos_w1 =
	    (word_embedding.dot(w1_embedding) + 1.0) / 2.0;
	double shifted_cos_w2 =
	    (word_embedding.dot(w2_embedding) + 1.0) / 2.0;
	double shifted_cos_v1 =
	    (word_embedding.dot(v1_embedding) + 1.0) / 2.0;
	double score =
	    shifted_cos_w2 * shifted_cos_v1 / (shifted_cos_w1 + 0.001);
	if (score > max_score) {
	    max_score = score;
	    predicted_v2 = word;
	}
    }
    ASSERT(!predicted_v2.empty(), "No answer for \"" << w1 << ":" << w2
	   << " as in " << v1 << ":" << "?\"");
    return predicted_v2;
}

void WordRep::PerformAgglomerativeClustering(size_t num_clusters) {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.exists(AgglomerativePath())) { return; }

    // Prepare a list of word vectors sorted in decreasing frequency.
    ASSERT(wordvectors_.size() > 0, "No word vectors to cluster!");
    vector<Eigen::VectorXd> sorted_vectors(sorted_wordcount_.size());
    for (size_t i = 0; i < sorted_wordcount_.size(); ++i) {
	string word_string = sorted_wordcount_[i].first;
	sorted_vectors[i] = wordvectors_[word_string];
    }

    // Do agglomerative clustering over the sorted word vectors.
    time_t begin_time_greedo = time(NULL);
    log_ << endl << "[Agglomerative clustering with " << num_clusters
	 << " clusters]" << endl;
    Greedo greedo;
    greedo.Cluster(sorted_vectors, num_clusters);
    double time_greedo = difftime(time(NULL), begin_time_greedo);
    StringManipulator string_manipulator;
    log_ << "   Average number of tightenings: "
	 << greedo.average_num_extra_tightening() << " (versus exhaustive "
	 << num_clusters << ")" << endl;
    log_ << "   Time taken: " << string_manipulator.print_time(time_greedo)
	 << endl;

    // Lexicographically sort bit strings for enhanced readability.
    vector<string> bitstring_types;
    for (const auto &bitstring_pair : *greedo.bit2cluster()) {
	bitstring_types.push_back(bitstring_pair.first);
    }
    sort(bitstring_types.begin(), bitstring_types.end());

    // Write the bit strings and their associated word types.
    ofstream greedo_file(AgglomerativePath(), ios::out);
    unordered_map<string, vector<size_t> > *bit2cluster = greedo.bit2cluster();
    for (const auto &bitstring : bitstring_types) {
	vector<pair<string, size_t> > sorting_vector;  // Sort each cluster.
	for (size_t cluster : bit2cluster->at(bitstring)) {
	    string word_string = sorted_wordcount_[cluster].first;
	    size_t count = sorted_wordcount_[cluster].second;
	    sorting_vector.push_back(make_pair(word_string, count));
	}
	sort(sorting_vector.begin(), sorting_vector.end(),
	     sort_pairs_second<string, size_t, greater<size_t> >());

	for (const auto &word_pair : sorting_vector) {
	    greedo_file << bitstring << " " << word_pair.first << " "
			<< word_pair.second << endl;
	}
    }
}

void WordRep::RotateWordVectorsToPCACoordinates() {
    ASSERT(wordvectors_.size() > 0, "No word vectors to rotate!");
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.exists(WordVectorsPCAPath())) { return; }

    time_t begin_time_pca = time(NULL);  // PCA time.
    Eigen::MatrixXd word_matrix(dim_, wordvectors_.size());
    for (size_t i = 0; i < wordvectors_.size(); ++i) {
	string word_string = word_num2str_[i];
	word_matrix.col(i) = wordvectors_[word_string];
    }
    log_ << endl << "[Change of basis to the PCA coordinates]" << endl << flush;
    for (size_t i = 0; i < dim_; ++i) {
	// Center each dimension (row) to have mean zero.
	double row_mean = word_matrix.row(i).mean();
	for (size_t j = 0; j < word_matrix.row(i).size(); ++j) {
	    word_matrix.row(i)(j) -= row_mean;
	}
    }

    // Do SVD on the centered word matrix and compute right singular vectors.
    Eigen::JacobiSVD<Eigen::MatrixXd> eigen_svd(word_matrix,
						Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = eigen_svd.singularValues();

    // Compute variance in each dimension.
    ofstream pca_variance_file(PCAVariancePath(), ios::out);
    vector<double> variances;
    double sum_variances = 0.0;
    for (size_t i = 0; i < singular_values.size(); ++i) {
	double ith_variance =
	    pow(singular_values(i), 2) / (word_matrix.cols() - 1);
	pca_variance_file << ith_variance << endl;
	variances.push_back(ith_variance);
	sum_variances += ith_variance;
    }
    double cumulative_variance = 0.0;
    const double variance_percentage = 90.0;
    for (size_t i = 0; i < variances.size(); ++i) {
	cumulative_variance += variances[i];
	if (cumulative_variance / sum_variances * 100 > variance_percentage) {
	    log_ << "   Top " << i + 1 << " PCA dimensions contain > "
		 << variance_percentage << "% of total variances" << endl;
	    break;
	}
    }

    // Compute word vectors in the PCA basis: right singular vectors times
    // scaled by singular values.
    Eigen::MatrixXd word_matrix_pca = eigen_svd.matrixV();
    word_matrix_pca.transposeInPlace();
    for (size_t i = 0; i < word_matrix_pca.rows(); ++i) {
	word_matrix_pca.row(i) *= singular_values(i);
    }
    double time_pca = difftime(time(NULL), begin_time_pca);
    StringManipulator string_manipulator;
    log_ << "   Time taken: " << string_manipulator.print_time(time_pca)
	 << endl;

    // Write word vectors in a PCA basis sorted in decreasing frequency.
    ofstream wordvectors_pca_file(WordVectorsPCAPath(), ios::out);
    for (size_t i = 0; i < sorted_wordcount_.size(); ++i) {
	string word_string = sorted_wordcount_[i].first;
	Word word = word_str2num_[word_string];
	size_t word_count = sorted_wordcount_[i].second;
	Eigen::VectorXd pca_vector = word_matrix_pca.col(word);
	wordvectors_pca_file << word_count << " " << word_string;
	for (size_t j = 0; j < pca_vector.size(); ++ j) {
	    wordvectors_pca_file << " " << pca_vector(j);
	}
	wordvectors_pca_file << endl;
    }
}

string WordRep::Signature(size_t version) {
    ASSERT(version <= 3, "Unrecognized signature version: " << version);

    string signature = "rare" + to_string(rare_cutoff_);
    if (version >= 1) {
	signature += "_window" + to_string(window_size_);
	if (bag_of_words_) {
	    signature += "_bow";
	}
	if (sentence_per_line_) {
	    signature += "_spl";
	}
    }
    if (version >= 2) {
	signature += "_dim" + to_string(dim_);
	signature += "_" + scaling_method_;
	if (scaling_method_ == "cca" || scaling_method_ == "scca" ||
	    scaling_method_ == "lcca" || scaling_method_ == "rreg") {
	    // Record the smoothing value for these scaling methods.
	    signature += "_smooth" + to_string(smooth_value_);
	}
    }
    if (version >= 3) {
	if (!weighting_method_.empty()) {
	    signature += "_" + weighting_method_ + "weight";
	}
    }
    return signature;
}
