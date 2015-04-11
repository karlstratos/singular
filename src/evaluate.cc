// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "evaluate.h"

#include "util.h"

void Evaluator::EvaluateWordSimilarity(
    const unordered_map<string, Eigen::VectorXd> &wordvectors,
    const string &file_path, size_t *num_instances, size_t *num_handled,
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
	string_manipulator.Split(line, " ", &tokens);
	ASSERT(tokens.size() == 3, "Wrong format for word similarity!");
	string word1 = tokens[0];
	string word2 = tokens[1];
	string word1_lowercase = string_manipulator.Lowercase(word1);
	string word2_lowercase = string_manipulator.Lowercase(word2);
	double human_score = stod(tokens[2]);

	// Get a vector for each word type. First, try to get a vector for the
	// original string. If not found, try lowercasing.
	Eigen::VectorXd vector_word1, vector_word2;
	if (wordvectors.find(word1) != wordvectors.end()) {
	    vector_word1 = wordvectors.at(word1);
	} else if (wordvectors.find(word1_lowercase) != wordvectors.end()) {
	    vector_word1 = wordvectors.at(word1_lowercase);
	}
	if (wordvectors.find(word2) != wordvectors.end()) {
	    vector_word2 = wordvectors.at(word2);
	} else if (wordvectors.find(word2_lowercase) != wordvectors.end()) {
	    vector_word2 = wordvectors.at(word2_lowercase);
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

void Evaluator::EvaluateWordAnalogy(const unordered_map<string, Eigen::VectorXd>
				    &wordvectors, const string &file_path,
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
	string_manipulator.Split(line, " ", &tokens);
	ASSERT(tokens.size() == 5, "Wrong format for word analogy!");
	// Ignore the analogy category: only compute the overall accuracy.
	string w1 = tokens[1];
	string w1_lowercase = string_manipulator.Lowercase(w1);
	string w2 = tokens[2];
	string w2_lowercase = string_manipulator.Lowercase(w2);
	string v1 = tokens[3];
	string v1_lowercase = string_manipulator.Lowercase(v1);
	string v2 = tokens[4];
	string v2_lowercase = string_manipulator.Lowercase(v2);
	analogies.push_back(make_tuple(w1, w2, v1, v2));

	// Get a vector for each word type. First, try to get a vector for the
	// original string. If not found, try lowercasing.
	if (wordvectors.find(w1) != wordvectors.end()) {
	    wordvectors_subset[w1] = wordvectors.at(w1);
	} else if (wordvectors.find(w1_lowercase) != wordvectors.end()) {
	    wordvectors_subset[w1] = wordvectors.at(w1_lowercase);
	}
	if (wordvectors.find(w2) != wordvectors.end()) {
	    wordvectors_subset[w2] = wordvectors.at(w2);
	} else if (wordvectors.find(w2_lowercase) != wordvectors.end()) {
	    wordvectors_subset[w2] = wordvectors.at(w2_lowercase);
	}
	if (wordvectors.find(v1) != wordvectors.end()) {
	    wordvectors_subset[v1] = wordvectors.at(v1);
	} else if (wordvectors.find(v1_lowercase) != wordvectors.end()) {
	    wordvectors_subset[v1] = wordvectors.at(v1_lowercase);
	}
	if (wordvectors.find(v2) != wordvectors.end()) {
	    wordvectors_subset[v2] = wordvectors.at(v2);
	} else if (wordvectors.find(v2_lowercase) != wordvectors.end()) {
	    wordvectors_subset[v2] = wordvectors.at(v2_lowercase);
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

string Evaluator::AnswerAnalogyQuestion(
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
