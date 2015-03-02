// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for inducing lexical representations.

#ifndef WORDREP_H
#define WORDREP_H

#include <fstream>

#include "decompose.h"

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

    // Removes the content in the output directory.
    void ResetOutputDirectory();

    // Extracts statistics from a corpus with preprocessing.
    void ExtractStatistics(const string &corpus_file);

    // Induces lexical representations from word counts in the output directory.
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

    // Sets the flag for weighting context dynamically.
    void set_dynamic_context_weight(bool dynamic_context_weight) {
	dynamic_context_weight_ = dynamic_context_weight;
    }

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

    // Sets the smoothing value.
    void set_smooth_value(size_t smooth_value) { smooth_value_ = smooth_value; }

    // Sets the weighting method.
    void set_weighting_method(string weighting_method) {
	weighting_method_ = weighting_method;
    }

    // Sets the maximum number of training epochs.
    void set_max_num_epochs(double max_num_epochs) {
	max_num_epochs_ = max_num_epochs;
    }

    // Sets the regularization term.
    void set_regularization_term(double regularization_term) {
	regularization_term_ = regularization_term;
    }

    // Sets the learning rate prior.
    void set_learning_rate_prior(double learning_rate_prior) {
	learning_rate_prior_ = learning_rate_prior;
    }

    // Returns the computed word vectors.
    unordered_map<string, Eigen::VectorXd> *wordvectors() {
	return &wordvectors_;
    }

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

    // Returns the integer ID corresponding to a word string.
    Word word_str2num(const string &word_string);

    // Returns the original string form of a word integer ID.
    string word_num2str(Word word);

    // Returns the integer ID corresponding to a context string.
    Context context_str2num(const string &context_string);

    // Returns the original string form of a context integer ID.
    string context_num2str(Context context);

private:
    // Extracts the count of each word type appearing in the given corpus file.
    void CountWords(const string &corpus_file);

    // Adds the word to the word dictionary if not already known.
    Word AddWordIfUnknown(const string &word_string);

    // Adds the context to the context dictionary if not already known.
    Context AddContextIfUnknown(const string &context_string);

    // Determines rare word types.
    void DetermineRareWords();

    // Slides a window across a corpus to collect statistics.
    void SlideWindow(const string &corpus_file);

    // Increments context counts according to how context is defined.
    void IncrementContextCount(
	const string &context_string, const string &position_string, Word word,
	size_t max_weight, size_t distance,
	unordered_map<Context, double> *count_context,
	unordered_map<Context, unordered_map<Word, double> >
	*count_word_context);

    // Induces vector representations of word types based on cached count files.
    void InduceWordVectors();

    // Loads a filtered word dictionary from a cached file.
    void LoadWordDictionary();

    // Load a sorted list of word-count pairs from a cached file.
    void LoadSortedWordCounts();

    // Calculate a word matrix (column = word) from cached count files.
    Eigen::MatrixXd CalculateWordMatrix();

    // Get an appropriately scaled weight matrix.
    SMat GetWeights(const string &weight_method);

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
    //    version=2: 1 + dim_, smooth_value_, transformation_method_,
    //                   scaling_method_
    //    version=3: 2 + weighting_method_
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

    // Returns the path to the word vectors.
    string WordVectorsPath() {
	return output_directory_ + "/wordvectors_" + Signature(3);
    }

    // Returns the path to the word vectors in a PCA basis.
    string WordVectorsPCAPath() {
	return output_directory_ + "/wordvectors_pca_" + Signature(3);
    }

    // Returns the path to the left singular vectors.
    string LeftSingularVectorsPath() {
	return output_directory_ + "/left_singular_vectors_" + Signature(2);
    }

    // Returns the path to the right singular vectors.
    string RightSingularVectorsPath() {
	return output_directory_ + "/right_singular_vectors_" + Signature(2);
    }

    // Returns the path to the singular values.
    string SingularValuesPath() {
	return output_directory_ + "/singular_values_" + Signature(2);
    }

    // Returns the path to the PCA variance values.
    string PCAVariancePath() {
	return output_directory_ + "/pca_variance_" + Signature(3);
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

    // Weight context dynamically?
    bool dynamic_context_weight_ = false;

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

    // Smoothing value.
    size_t smooth_value_ = 0;

    // Weighting method.
    string weighting_method_ = "";

    // Maximum number of training epochs.
    size_t max_num_epochs_ = 100;

    // Regularization term.
    double regularization_term_ = 0.1;

    // Learning rate prior.
    double learning_rate_prior_ = 0.1;
};

#endif  // WORDREP_H
