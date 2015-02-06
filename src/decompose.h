// Author: Karl Stratos (karlstratos@gmail.com)
//
// Code for scaling and decomposing matrices.

#ifndef DECOMPOSE_H
#define DECOMPOSE_H

#include <Eigen/Dense>

#include "sparsesvd.h"

class Decomposer {
public:
    // Initializes empty.
    Decomposer() { }

    // Initializes with a dimension.
    Decomposer(size_t dim) : dim_(dim) { }

    ~Decomposer() { }

    // Decomposes a matrix with scaling.
    void Decompose(unordered_map<size_t, unordered_map<size_t, double> >
		   *joint_values,
		   const unordered_map<size_t, double> &values1,
		   const unordered_map<size_t, double> &values2);

    // Loads a matrix and scaling values from files and performs decomposition.
    void Decompose(const string &joint_values_path, const string &values1_path,
		   const string &values2_path);

    // Computes values from samples and performs decomposition.
    void Decompose(const vector<unordered_map<size_t, double> > &samples1,
		   const vector<unordered_map<size_t, double> > &samples2);

    // Set the file containing left singular vectors (as rows).
    void set_left_singular_vectors_path(string left_singular_vectors_path) {
	left_singular_vectors_path_ = left_singular_vectors_path;
    }

    // Set the file containing right singular vectors (as rows).
    void set_right_singular_vectors_path(string right_singular_vectors_path) {
	right_singular_vectors_path_ = right_singular_vectors_path;
    }

    // Set the file containing singular values.
    void set_singular_values_path(string singular_values_path) {
	singular_values_path_ = singular_values_path;
    }

    // Sets the number of samples.
    void set_num_samples(size_t num_samples) { num_samples_ = num_samples; }

    // Sets a target dimension.
    void set_dim(size_t dim) { dim_ = dim; }

    // Sets the scaling method.
    void set_scaling_method(string scaling_method) {
	scaling_method_ = scaling_method;
    }

    // Sets a smoothing value.
    void set_smooth_value(size_t smooth_value) { smooth_value_ = smooth_value; }

    // Sets the weight matrix.
    void set_weights(SMat weights) { weights_ = weights; }

    // Sets the regularization term.
    void set_regularization_term(double regularization_term) {
	regularization_term_ = regularization_term;
    }

    // Sets the learning rate prior.
    void set_learning_rate_prior(double learning_rate_prior) {
	learning_rate_prior_ = learning_rate_prior;
    }

    // Returns a matrix of scaled left singular vectors (as rows).
    Eigen::MatrixXd *left_matrix() { return &left_matrix_; }

    // Returns a matrix of scaled right singular vectors (as rows).
    Eigen::MatrixXd *right_matrix() { return &right_matrix_; }

    // Returns singular values.
    Eigen::VectorXd *singular_values() { return &singular_values_; }

    // Returns the number of samples.
    size_t num_samples() { return num_samples_; }

    // Returns the target dimension.
    size_t dim() { return dim_; }

    // Returns the actually computed SVD rank (might be smaller than the target
    // dimension if rank(matrix) < target dimension).
    size_t rank() { return rank_; }

    // Returns the scaling method.
    string scaling_method() { return scaling_method_; }

    // Returns the smoothing value.
    size_t smooth_value() { return smooth_value_; }

    // Returns the weight matrix.
    SMat weights() { return weights_; }

private:
    // Computes SVD with the given solver if no specified cache can be found.
    void ComputeSVDIfNecessary(SparseSVDSolver *svd_solver);

    // Loads scaling values from a file.
    void LoadScalingValues(const string &scaling_values_path,
			   unordered_map<size_t, double> *scaling_values);

    // Scales a joint value by individual values.
    double ScaleJointValue(double joint_value, double value1, double value2);

    // Extracts scaled singular vectors from an SVD solver.
    void ExtractFromSVD(SparseSVDSolver *svd_solver,
			const unordered_map<size_t, double> &values1,
			const unordered_map<size_t, double> &values2);

    // Scales a value in a matrix by given row and column values.
    double ScaleMatrixValue(double matrix_value, double row_value,
			    double column_value);

    // Matrix of scaled left singular vectors (as rows).
    Eigen::MatrixXd left_matrix_;

    // Matrix of scaled right singular vectors (as rows).
    Eigen::MatrixXd right_matrix_;

    // Singular values.
    Eigen::VectorXd singular_values_;

    // File containing left singular vectors (as rows).
    string left_singular_vectors_path_ = "";

    // File containing right singular vectors (as rows).
    string right_singular_vectors_path_ = "";

    // File containing singular values.
    string singular_values_path_ = "";

    // Number of samples.
    size_t num_samples_ = 0;

    // Target dimension.
    size_t dim_ = 0;

    // Actually computed SVD rank (might be smaller than the target dimension
    // if rank(matrix) < target dimension).
    size_t rank_ = 0;

    // Scaling method.
    string scaling_method_ = "cca";

    // Smoothing value.
    size_t smooth_value_ = 5;

    // Weight matrix.
    SMat weights_ = nullptr;

    // Regularization term.
    double regularization_term_ = 0.1;

    // Learning rate prior.
    double learning_rate_prior_ = 0.1;
};

#endif  // DECOMPOSE_H
