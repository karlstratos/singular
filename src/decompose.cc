// Author: Karl Stratos (karlstratos@gmail.com)

#include "decompose.h"

#include <fstream>

#include "wsqloss.h"

void Decomposer::Decompose(
    unordered_map<size_t, unordered_map<size_t, double> > *joint_values,
    const unordered_map<size_t, double> &values1,
    const unordered_map<size_t, double> &values2) {
    // Scale the joint values by individual values.
    for (const auto &col_pair : *joint_values) {
	size_t col = col_pair.first;
	for (const auto &row_pair : col_pair.second) {
	    size_t row = row_pair.first;
	    (*joint_values)[col][row] =
		ScaleJointValue(joint_values->at(col)[row],
				 values1.at(row), values2.at(col));
	}
    }

    // Perform an SVD on the scaled joint values.
    SparseSVDSolver svd_solver;
    svd_solver.LoadSparseMatrix(*joint_values);
    ComputeSVDIfNecessary(&svd_solver);

    // Extract scaled singular vectors from the SVD.
    ExtractFromSVD(&svd_solver, values1, values2);
}

void Decomposer::Decompose(const string &joint_values_path,
			   const string &values1_path,
			   const string &values2_path) {
    // Load a matrix of joint values directly into an SVD solver.
    SparseSVDSolver svd_solver(joint_values_path);

    // Load first scaling values.
    unordered_map<size_t, double> values1;
    LoadScalingValues(values1_path, &values1);

    // Load second scaling values.
    unordered_map<size_t, double> values2;
    LoadScalingValues(values2_path, &values2);

    // Check that dimensions match.
    SMat matrix = svd_solver.sparse_matrix();
    ASSERT(matrix->rows == values1.size() &&
	   matrix->cols == values2.size(), "Dimensions don't match: "
	   << values1.size() << "x1, " << matrix->rows << "x" << matrix->cols
	   << ", " << values2.size() << "x1");

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

    // Extract scaled singular vectors from the SVD.
    ExtractFromSVD(&svd_solver, values1, values2);
}

void Decomposer::Decompose(
    const vector<unordered_map<size_t, double> > &samples1,
    const vector<unordered_map<size_t, double> > &samples2) {
    ASSERT(samples1.size() == samples2.size(), "Sample sequences need to "
	   "have the same length.");
    num_samples_ = samples1.size();

    // Compute unnormalized covariance values from the samples. Note that this
    // degenerates to "counting" if samples are zero-one vectors.
    unordered_map<size_t, unordered_map<size_t, double> > joint_values;
    unordered_map<size_t, double> values1;
    unordered_map<size_t, double> values2;
    for (size_t sample_num = 0; sample_num < samples1.size(); ++sample_num) {
	for (const auto &index1_value1_pair : samples1[sample_num]) {
	    size_t index1 = index1_value1_pair.first;
	    double value1 = index1_value1_pair.second;
	    values1[index1] += pow(value1, 2);  // Variance of 0-mean variable.
	}
	for (const auto &index2_value2_pair : samples2[sample_num]) {
	    size_t index2 = index2_value2_pair.first;
	    double value2 = index2_value2_pair.second;
	    values2[index2] += pow(value2, 2);  // Variance of 0-mean variable.
	    for (const auto &index1_value1_pair : samples1[sample_num]) {
		size_t index1 = index1_value1_pair.first;
		double value1 = index1_value1_pair.second;
		// This computes the covariance of 0-mean variables.
		joint_values[index2][index1] += value1 * value2;
	    }
	}
    }

    // Decompose a matrix of joint values with scaling.
    Decompose(&joint_values, values1, values2);
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
	    (file_manipulator.exists(left_singular_vectors_path_) &&
	     file_manipulator.exists(right_singular_vectors_path_) &&
	     file_manipulator.exists(singular_values_path_));
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
	    file_manipulator.write(left_matrix_, left_singular_vectors_path_);
	    file_manipulator.write(right_matrix_, right_singular_vectors_path_);
	    file_manipulator.write(singular_values_, singular_values_path_);
	}
    } else {  // We have located a cache, just read this SVD result.
	file_manipulator.read(left_singular_vectors_path_, &left_matrix_);
	file_manipulator.read(right_singular_vectors_path_, &right_matrix_);
	file_manipulator.read(singular_values_path_, &singular_values_);
	ASSERT(left_matrix_.rows() == right_matrix_.rows() &&
	       right_matrix_.rows() == singular_values_.size(),
	       "Loaded SVD result has faulty dimensions");
	rank_ = 0;
	for (size_t i = 0; i < singular_values_.size(); ++i) {
	    if (singular_values_(i) > 0) { ++rank_; }
	}
    }
}

void Decomposer::LoadScalingValues(
    const string &scaling_values_path,
    unordered_map<size_t, double> *scaling_values) {
    ifstream file(scaling_values_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << scaling_values_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    scaling_values->clear();
    size_t i = 0;
    while (file.good()) {
	getline(file, line);  // Scaling value for the i-th dimension.
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	ASSERT(tokens.size() == 1, "Bad format: " << line);
	(*scaling_values)[i++] = stod(tokens[0]);
    }
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
    } else if (transformation_method_ == "anscombe") {  // Anscombe.
	joint_value = 2.0 * sqrt(joint_value + 0.375);
	value1 = 2.0 * sqrt(value1 + 0.375);
	value2 = 2.0 * sqrt(value2 + 0.375);
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
	scaled_joint_value /= sqrt(value1 + smooth_value_);
	scaled_joint_value /= sqrt(value2 + smooth_value_);
    } else if (scaling_method_ == "rreg") {
	// Ridge regression scaling.
	scaled_joint_value /= value1 + smooth_value_;
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

void Decomposer::ExtractFromSVD(SparseSVDSolver *svd_solver,
				const unordered_map<size_t, double> &values1,
				const unordered_map<size_t, double> &values2) {
    // Ensure that we have loaded the SVD result.
    ASSERT(left_matrix_.rows() > 0 && left_matrix_.cols() > 0 &&
	   right_matrix_.rows() > 0 && right_matrix_.cols() > 0 &&
	   singular_values_.size() > 0, "No SVD result for extraction.");

    // Check that dimensions match for scaling.
    size_t dim1 = values1.size();
    size_t dim2 = values2.size();
    ASSERT(left_matrix_.rows() == dim_ && left_matrix_.cols() == dim1 &&
	   right_matrix_.rows() == dim_ && right_matrix_.cols() == dim2,
	   "Dimensions don't match between the SVD result and scaling values.");

    // If weights are given, perform weighted squared loss minimization on top
    // of SVD and return.
    if (weights_ != nullptr) {
	WSQLossOptimizer wsq_loss_optimizer;
	wsq_loss_optimizer.set_max_num_epochs(max_num_epochs_);
	wsq_loss_optimizer.set_regularization_term(regularization_term_);
	wsq_loss_optimizer.set_learning_rate_prior(learning_rate_prior_);
	SMat values = svd_solver->sparse_matrix();

	// Multiply the left singular vectors by singular values to use the
	// singular values in the SVD decomposition.
	for (size_t row = 0; row < dim_; ++row) {
	    left_matrix_.row(row) *= singular_values_(row);
	}

	// Initialize the optimization process with SVD.
	wsq_loss_optimizer.Optimize(weights_, values, &left_matrix_,
				    &right_matrix_);
	return;  // Don't do post-SVD scaling.
    }

    // Otherwise, do post-SVD singular vector scaling.
    for (size_t row = 0; row < dim_; ++row) {
	for (size_t col = 0; col < dim1; ++col) {
	    left_matrix_(row, col) = ScaleMatrixValue(left_matrix_(row, col),
						      singular_values_(row),
						      values1.at(col));
	}
	for (size_t col = 0; col < dim2; ++col) {
	    right_matrix_(row, col) = ScaleMatrixValue(right_matrix_(row, col),
						       singular_values_(row),
						       values2.at(col));
	}
    }
}

double Decomposer::ScaleMatrixValue(double matrix_value, double row_value,
				    double column_value) {
    double scaled_matrix_value = matrix_value;
    if (scaling_method_ == "cca") {
	scaled_matrix_value /= sqrt(column_value + smooth_value_);
    } else if (scaling_method_ == "raw" || scaling_method_ == "rreg") {
	scaled_matrix_value *= row_value;
    } else if (scaling_method_ == "ppmi") {
	scaled_matrix_value *= sqrt(row_value);
    } else {
	ASSERT(false, "Unknown scaling method: " << scaling_method_);
    }
    return scaled_matrix_value;
}
