// Author: Karl Stratos (stratos@cs.columbia.edu)
//
//  Code for inducing lexical representations.

#ifndef WORDREP_H
#define WORDREP_H

#include <Eigen/Dense>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef size_t Word;
typedef size_t Context;

class WordRep {
public:
    // Initializes empty.
    WordRep() { }

    // Initializes with an output directory.
    WordRep(const string &output_directory) {
	SetOutputDirectory(output_directory);
    }

    ~WordRep() { }

    // Sets the output directory.
    void SetOutputDirectory(const string &output_directory);

    // Resets the content in the output directory.
    void ResetOutputDirectory();

    // Extracts statistics from a corpus (file or a directory of files).
    void ExtractStatistics(const string &corpus_file);

    // Induces lexical representations from cached word counts.
    void InduceLexicalRepresentations();

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

    // Sets the target dimension of word vectors.
    void set_dim(size_t dim) { dim_ = dim; }

    // Sets the flag for smoothing context counts.
    void set_context_smoothing(bool context_smoothing) {
	context_smoothing_ = context_smoothing;
    }

    // Sets the transformation method.
    void set_transformation_method(string transformation_method) {
	transformation_method_ = transformation_method;
    }

    // Sets the scaling method.
    void set_scaling_method(string scaling_method) {
	scaling_method_ = scaling_method;
    }

    // Sets the weighting method.
    void set_weighting_method(string weighting_method) {
	weighting_method_ = weighting_method;
    }

    // Returns the computed word vectors.
    unordered_map<string, Eigen::VectorXd> *wordvectors() {
	return &wordvectors_;
    }

    // Sets the maximum number of training epochs.
    void set_max_num_epochs(size_t max_num_epochs) {
	max_num_epochs_ = max_num_epochs;
    }

    // Sets the number of threads.
    void set_num_threads(size_t num_threads) { num_threads_ = num_threads; }

    // Returns the singular values of the scaled count matrix.
    Eigen::VectorXd *singular_values() { return &singular_values_; }

    // Returns the special string for representing rare words.
    string kRareString() { return kRareString_; }

    // Returns the special string for buffering.
    string kBufferString() { return kBufferString_; }

    // Returns the path to the word-context count file.
    string CountWordContextPath() {
	return output_directory_ + "/count_word_context_" + Signature(1);
    }

    // Returns the path to the word count file.
    string CountWordPath() {
	return output_directory_ + "/count_word_" + Signature(0);
    }

    // Returns the path to the context count file.
    string CountContextPath() {
	return output_directory_ + "/count_context_" + Signature(1);
    }

    // Loads a filtered word dictionary from a cached file.
    void LoadWordDictionary();

    // Loads a filtered context dictionary from a cached file.
    void LoadContextDictionary();

    // Returns the integer ID corresponding to a word string.
    Word word_str2num(const string &word_string);

    // Returns the original string form of a word integer ID.
    string word_num2str(Word word);

    // Returns the integer ID corresponding to a context string.
    Context context_str2num(const string &context_string);

    // Returns the original string form of a context integer ID.
    string context_num2str(Context context);

private:
    // Extracts the count of each word type appearing in the given corpus.
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

    // Induces vector representations of word types based on cached count files.
    void InduceWordVectors();

    // Load a sorted list of word-count pairs from a cached file.
    void LoadSortedWordCounts();

    // Calculate SVD of cached count files.
    void CalculateSVD();

    // Scales a joint value by individual values.
    double ScaleJointValue(double joint_value, double value1, double value2,
			   size_t num_samples);

    // Calculate weighted least squares of cached count files.
    void CalculateWeightedLeastSquares();

    // Tests the quality of word vectors on simple tasks.
    void TestQualityOfWordVectors();

    // Evaluate word vectors on a word similarity dataset.
    void EvaluateWordSimilarity(const string &file_path,
				size_t *num_instances, size_t *num_handled,
				double *correlation);

    // Evaluate word vectors on a word analogy dataset.
    void EvaluateWordAnalogy(const string &file_path,
			     size_t *num_instances, size_t *num_handled,
			     double *accuracy);

    // Returns word v2 (not in {w1, w2, v1}) such that "w1:w2 as in v1:v2".
    string AnswerAnalogyQuestion(
	string w1, string w2, string v1,
	const unordered_map<string, Eigen::VectorXd> &wordvectors_subset);

    // Performs greedy agglomerative clustering over word vectors.
    void PerformAgglomerativeClustering(size_t num_clusters);

    // Performs PCA on word vectors to put them in the PCA basis.
    void RotateWordVectorsToPCACoordinates();

    // Returns a string signature of tunable parameters.
    //    version=0: rare_cutoff_
    //    version=1: 0 + sentence_per_line_, window_size_, context_defintion_
    //    version=2: 1 + dim_, transformation_method_, scaling_method_
    string Signature(size_t version);

    // Returns the path to the corpus information file.
    string CorpusInfoPath() { return output_directory_ + "/corpus_info"; }

    // Returns the path to the log file.
    string LogPath() { return output_directory_ + "/log"; }

    // Returns the path to the sorted word types file.
    string SortedWordTypesPath() {
	return output_directory_ + "/sorted_word_types";
    }

    // Returns the path to the rare word file.
    string RarePath() {
	return output_directory_ + "/rare_words_" + Signature(0);
    }

    // Returns the path to the str2num mapping for words.
    string WordStr2NumPath() {
	return output_directory_ + "/word_str2num_" + Signature(0);
    }

    // Returns the path to the str2num mapping for context.
    string ContextStr2NumPath() {
	return output_directory_ + "/context_str2num_" + Signature(1);
    }

    // Returns the path to the singular values.
    string SingularValuesPath() {
	return output_directory_ + "/singular_values_" + Signature(2);
    }

    // Returns the path to the word vectors.
    string WordVectorsPath() {
	return output_directory_ + "/wordvectors_" + Signature(3);
    }

    // Returns the path to the agglomeratively clusterered word vectors.
    string AgglomerativePath() {
	return output_directory_ + "/agglomerative_" + Signature(3);
    }

    // Word-count pairs sorted in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount_;

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

    // Computed word vectors.
    unordered_map<string, Eigen::VectorXd> wordvectors_;

    // Matrix of word vectors (as rows).
    Eigen::MatrixXd word_matrix_;

    // Matrix of context vectors (as rows).
    Eigen::MatrixXd context_matrix_;

    // Singular values of the correlation matrix.
    Eigen::VectorXd singular_values_;

    // Path to the output directory.
    string output_directory_;

    // If a word type appears <= this number, treat it as a rare symbol.
    size_t rare_cutoff_ = 1;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the context to compute covariance on. Note that it needs to be
    // odd if we want the left and right context to have the same length.
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "bag";

    // Target dimension of word vectors.
    size_t dim_;

    // Smooth context counts?
    bool context_smoothing_ = false;

    // Data transformation method.
    string transformation_method_ = "raw";

    // Scaling method.
    string scaling_method_ = "cca";

    // Weighting method.
    string weighting_method_ = "freq";

    // Maximum number of training epochs.
    size_t max_num_epochs_ = 100;

    // Number of threads.
    size_t num_threads_ = 20;
};

#endif  // WORDREP_H
