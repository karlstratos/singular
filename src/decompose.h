// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for scaling and decomposing sparse matrices.

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

    // Loads a matrix and scaling values from files and performs decomposition.
    void Decompose(const string &joint_values_path, const string &values1_path,
		   const string &values2_path);

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

private:
    // Scales a joint value by individual values.
    double ScaleJointValue(double joint_value, double value1, double value2);

    // Computes SVD with the given solver if no specified cache can be found.
    void ComputeSVDIfNecessary(SparseSVDSolver *svd_solver);

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

    // Smooth context counts?
    bool context_smoothing_ = false;

    // Data transformation method.
    string transformation_method_ = "sqrt";

    // Scaling method.
    string scaling_method_ = "cca";
};

#endif  // DECOMPOSE_H
