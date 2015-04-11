// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for evaluation.

#ifndef EVALUATE_H
#define EVALUATE_H

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Evaluator {
public:
    // Evaluate word vectors on a word similarity dataset.
    void EvaluateWordSimilarity(const unordered_map<string, Eigen::VectorXd>
				&wordvectors, const string &file_path,
				size_t *num_instances, size_t *num_handled,
				double *correlation);

    // Evaluate word vectors on a word analogy dataset.
    void EvaluateWordAnalogy(const unordered_map<string, Eigen::VectorXd>
			     &wordvectors, const string &file_path,
			     size_t *num_instances, size_t *num_handled,
			     double *accuracy);

    // Returns word v2 (not in {w1, w2, v1}) such that "w1:w2 ~ v1:v2".
    string AnswerAnalogyQuestion(string w1, string w2, string v1,
				 const unordered_map<string, Eigen::VectorXd>
				 &wordvectors_subset);
};

#endif  // EVALUATE_H
