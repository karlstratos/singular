// Author: Karl Stratos (karlstratos@gmail.com)

#include "decompose.h"

#include <fstream>

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
    if (dim_ == 0) { dim_ = min(values1.size(), values2.size()); }
    svd_solver.SolveSparseSVD(dim_);
    rank_ = svd_solver.rank();

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
    if (dim_ == 0) { dim_ = min(matrix->rows, matrix->cols); }
    svd_solver.SolveSparseSVD(dim_);
    rank_ = svd_solver.rank();

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
    double scaled_joint_value = joint_value;
    if (scaling_method_ == "cca") {
	scaled_joint_value /= sqrt(value1 + smooth_value_);
	scaled_joint_value /= sqrt(value2 + smooth_value_);
    } else if (scaling_method_ == "ppmi") {
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
    size_t dim1 = values1.size();
    size_t dim2 = values2.size();
    ASSERT(svd_solver->HasSVDResult(), "No SVD result for extraction.");
    ASSERT(svd_solver->left_singular_vectors()->rows == dim_ &&
	   svd_solver->left_singular_vectors()->cols == dim1 &&
	   svd_solver->right_singular_vectors()->rows == dim_ &&
	   svd_solver->right_singular_vectors()->cols == dim2,
	   "Dimensions don't match between the SVD result and scaling values.");

    // Collect singular values.
    singular_values_.resize(dim_);
    for (size_t i = 0; i < dim_; ++i) {
	singular_values_(i) = *(svd_solver->singular_values() + i);
    }

    // Collect a matrix of left singular vectors as rows.
    left_matrix_.resize(dim_, dim1);
    for (size_t row = 0; row < dim_; ++row) {
	for (size_t col = 0; col < dim1; ++col) {
	    left_matrix_(row, col) =
		svd_solver->left_singular_vectors()->value[row][col];
	}
    }

    // Collect a matrix of right singular vectors as rows.
    right_matrix_.resize(dim_, dim2);
    for (size_t row = 0; row < dim_; ++row) {
	for (size_t col = 0; col < dim2; ++col) {
	    right_matrix_(row, col) =
		svd_solver->right_singular_vectors()->value[row][col];
	}
    }
    svd_solver->FreeSVDResult();  // We have the SVD result: free the memory.

    // TODO: Compute weighted decomposition here.

    // Post-SVD singular vector scaling.
    for (size_t row = 0; row < dim_; ++row) {
	for (size_t col = 0; col < dim1; ++col) {  // Left singular vectors.
	    if (scaling_method_ == "cca") {
		left_matrix_(row, col) /= sqrt(values1.at(col) + smooth_value_);
	    } else if (scaling_method_ == "ppmi") {
		left_matrix_(row, col) *= sqrt(singular_values_(row));
	    } else {
		ASSERT(false, "Unknown scaling method: " << scaling_method_);
	    }
	}
	for (size_t col = 0; col < dim2; ++col) {  // Right singular vectors.
	    if (scaling_method_ == "cca") {
		right_matrix_(row, col) /= sqrt(values2.at(col) +
						smooth_value_);
	    } else if (scaling_method_ == "ppmi") {
		right_matrix_(row, col) *= sqrt(singular_values_(row));
	    } else {
		ASSERT(false, "Unknown scaling method: " << scaling_method_);
	    }
	}
    }
}
