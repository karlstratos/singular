// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "decompose.h"

#include <fstream>

void Decomposer::Decompose(const string &joint_values_path,
			   const string &values1_path,
			   const string &values2_path) {
    // Load a sparse matrix of joint values directly into an SVD solver.
    SparseSVDSolver svd_solver(joint_values_path);

    // Load individual scaling values.
    FileManipulator file_manipulator;
    unordered_map<size_t, double> values1;
    unordered_map<size_t, double> values2;
    file_manipulator.Read(values1_path, &values1);
    file_manipulator.Read(values2_path, &values2);

    SMat matrix = svd_solver.sparse_matrix();
    ASSERT(matrix->rows == values1.size() && matrix->cols == values2.size(),
	   "Dimensions don't match: " << values1.size() << "x1, "
	   << matrix->rows << "x" << matrix->cols << ", " << values2.size()
	   << "x1");

    // Scale the joint values by individual values.
    for (size_t col = 0; col < matrix->cols; ++col) {
	size_t current_column_nonzero_index = matrix->pointr[col];
	size_t next_column_start_nonzero_index = matrix->pointr[col + 1];
	while (current_column_nonzero_index < next_column_start_nonzero_index) {
	    size_t row = matrix->rowind[current_column_nonzero_index];
	    matrix->value[current_column_nonzero_index] =
		ScaleJointValue(matrix->value[current_column_nonzero_index],
				values1[row], values2[col]);
	    ++current_column_nonzero_index;
	}
    }

    // Perform an SVD on the scaled joint values.
    ComputeSVDIfNecessary(&svd_solver);
}

double Decomposer::ScaleJointValue(double joint_value,
				   double value1, double value2) {
    if (context_smoothing_) {  // Context smoothing.
	value2 = pow(value2, 0.75);
    }

    // Data transformation.
    if (transformation_method_ == "raw") {  // No transformation.
    } else if (transformation_method_ == "sqrt") {  // Take square-root.
	joint_value = sqrt(joint_value);
	value1 = sqrt(value1);
	value2 = sqrt(value2);
    } else if (transformation_method_ == "two-thirds") {  // Power of 2/3.
	double power = 2.0 / 3.0;
	joint_value = pow(joint_value, power);
	value1 = pow(value1, power);
	value2 = pow(value2, power);
    } else if (transformation_method_ == "log") {  // Take log.
	joint_value = log(1.0 + joint_value);
	value1 = log(1.0 + value1);
	value2 = log(1.0 + value2);
    } else {
	ASSERT(false, "Unknown data transformation method: "
	       << transformation_method_);
    }

    // Scale the joint value by individual values (or not).
    double scaled_joint_value = joint_value;
    if (scaling_method_ == "raw") {  // No scaling.
    } else if (scaling_method_ == "cca") {
	// Canonical correlation analysis scaling.
	scaled_joint_value /= sqrt(value1);
	scaled_joint_value /= sqrt(value2);
    } else if (scaling_method_ == "reg") {
	// Ridge regression scaling.
	scaled_joint_value /= value1;
    } else if (scaling_method_ == "ppmi") {
	// Positive pointwise mutual information scaling.
	if (scaling_method_ == "ppmi") {
	    ASSERT(num_samples_ > 0, "Need the number of samples for PPMI");
	}
	scaled_joint_value = log(scaled_joint_value);
	scaled_joint_value += log(num_samples_);
	scaled_joint_value -= log(value1);
	scaled_joint_value -= log(value2);
	scaled_joint_value = max(scaled_joint_value, 0.0);
    } else {
	ASSERT(false, "Unknown scaling method: " << scaling_method_);
    }
    return scaled_joint_value;
}

void Decomposer::ComputeSVDIfNecessary(SparseSVDSolver *svd_solver) {
    // Are we given a pointer to a stored SVD result?
    bool cache_is_specified = (!left_singular_vectors_path_.empty() &&
			       !right_singular_vectors_path_.empty() &&
			       !singular_values_path_.empty());

    // Can we actually locate that SVD result?
    bool cache_is_found = false;
    FileManipulator file_manipulator;
    if (cache_is_specified) {
	cache_is_found =
	    (file_manipulator.Exists(left_singular_vectors_path_) &&
	     file_manipulator.Exists(right_singular_vectors_path_) &&
	     file_manipulator.Exists(singular_values_path_));
    }

    if (!cache_is_found) {  // We cannot locate a cache.
	size_t dim1 = svd_solver->sparse_matrix()->rows;
	size_t dim2 = svd_solver->sparse_matrix()->cols;
	if (dim_ == 0) { dim_ = min(dim1, dim2); }
	svd_solver->SolveSparseSVD(dim_);  // Computing SVD here!
	rank_ = svd_solver->rank();

	// Load a matrix of left singular vectors as rows.
	left_matrix_.resize(dim_, dim1);
	for (size_t row = 0; row < dim_; ++row) {
	    for (size_t col = 0; col < dim1; ++col) {
		left_matrix_(row, col) =
		    svd_solver->left_singular_vectors()->value[row][col];
	    }
	}

	// Load a matrix of right singular vectors as rows.
	right_matrix_.resize(dim_, dim2);
	for (size_t row = 0; row < dim_; ++row) {
	    for (size_t col = 0; col < dim2; ++col) {
		right_matrix_(row, col) =
		    svd_solver->right_singular_vectors()->value[row][col];
	    }
	}

	// Load singular values.
	singular_values_.resize(dim_);
	for (size_t i = 0; i < dim_; ++i) {
	    singular_values_(i) = *(svd_solver->singular_values() + i);
	}
	svd_solver->FreeSVDResult();  // Free memory.

	if (cache_is_specified) {
	    // Store this result to a specified location.
	    file_manipulator.Write(left_matrix_, left_singular_vectors_path_);
	    file_manipulator.Write(right_matrix_, right_singular_vectors_path_);
	    file_manipulator.Write(singular_values_, singular_values_path_);
	}
    } else {  // We have located a cache, just read this SVD result.
	file_manipulator.Read(left_singular_vectors_path_, &left_matrix_);
	file_manipulator.Read(right_singular_vectors_path_, &right_matrix_);
	file_manipulator.Read(singular_values_path_, &singular_values_);
	ASSERT(left_matrix_.rows() == right_matrix_.rows() &&
	       right_matrix_.rows() == singular_values_.size(),
	       "Loaded SVD result has faulty dimensions");
	rank_ = 0;
	for (size_t i = 0; i < singular_values_.size(); ++i) {
	    if (singular_values_(i) > 0) { ++rank_; }
	}
    }
}
