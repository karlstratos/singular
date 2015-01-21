// Author: Karl Stratos (karlstratos@gmail.com)

#include "sparsecca.h"

#include <fstream>

void SparseCCASolver::PerformCCA(
    unordered_map<size_t, unordered_map<size_t, double> > *covariance_xy,
    const unordered_map<size_t, double> &variance_x,
    const unordered_map<size_t, double> &variance_y) {
    if (scaling_method_ == "pmi") {
	ASSERT(num_samples_ > 0, "Need the number of samples for PMI");
    }

    // Scale the cross-covariance matrix. This corresponds to an approximate
    // whitening of the data.
    for (const auto &col_pair : *covariance_xy) {
	size_t col = col_pair.first;
	for (const auto &row_pair : col_pair.second) {
	    size_t row = row_pair.first;
	    (*covariance_xy)[col][row] =
		ScaleCovariance(covariance_xy->at(col)[row],
				variance_x.at(row), variance_y.at(col));
	}
    }

    // Perform an SVD on the scaled cross-covariance matrix.
    SparseSVDSolver svd_solver;
    svd_solver.LoadSparseMatrix(*covariance_xy);
    if (cca_dim_ == 0) { cca_dim_ = min(variance_x.size(), variance_y.size()); }
    svd_solver.SolveSparseSVD(cca_dim_);
    rank_ = svd_solver.rank();

    // Extract CCA parameters from the SVD result.
    ExtractCCAProjections(svd_solver, variance_x, variance_y);
}

void SparseCCASolver::PerformCCA(const string &covariance_xy_file,
				 const string &variance_x_file,
				 const string &variance_y_file) {
    if (scaling_method_ == "pmi") {
	ASSERT(num_samples_ > 0, "Need the number of samples for PMI");
    }

    // Load the variance for view X.
    unordered_map<size_t, double> variance_x;
    LoadVariance(variance_x_file, &variance_x);

    // Load the variance for view Y.
    unordered_map<size_t, double> variance_y;
    LoadVariance(variance_y_file, &variance_y);

    // Load the cross-covariance matrix.
    SparseSVDSolver svd_solver(covariance_xy_file);
    SMat matrix = svd_solver.sparse_matrix();
    ASSERT(matrix->rows == variance_x.size() &&
	   matrix->cols == variance_y.size(), "Dimensions don't match: "
	   << variance_x.size() << "x1, " << matrix->rows << "x" << matrix->cols
	   << ", " << variance_y.size() << "x1");

    // Scale the cross-covariance matrix. This corresponds to an approximate
    // whitening of the data.
    for (size_t col = 0; col < matrix->cols; ++col) {
	size_t current_column_nonzero_index = matrix->pointr[col];
	size_t next_column_start_nonzero_index = matrix->pointr[col + 1];
	while (current_column_nonzero_index < next_column_start_nonzero_index) {
	    size_t row = matrix->rowind[current_column_nonzero_index];
	    matrix->value[current_column_nonzero_index] =
		ScaleCovariance(matrix->value[current_column_nonzero_index],
				variance_x[row], variance_y[col]);
	    ++current_column_nonzero_index;
	}
    }

    // Perform an SVD on the scaled cross-covariance matrix.
    if (cca_dim_ == 0) { cca_dim_ = min(matrix->rows, matrix->cols); }
    svd_solver.SolveSparseSVD(cca_dim_);
    rank_ = svd_solver.rank();

    // Extract CCA parameters from the SVD result.
    ExtractCCAProjections(svd_solver, variance_x, variance_y);
}

void SparseCCASolver::PerformCCA(
    const vector<unordered_map<size_t, double> > &examples_x,
    const vector<unordered_map<size_t, double> > &examples_y) {
    ASSERT(examples_x.size() == examples_y.size(), "Example sequences need to "
	   "have the same length.");
    num_samples_ = examples_x.size();

    // Compute unnormalized covariance values from the examples.
    unordered_map<size_t, unordered_map<size_t, double> > covariance_xy;
    unordered_map<size_t, double> variance_x;
    unordered_map<size_t, double> variance_y;
    for (size_t example_num = 0; example_num < examples_x.size();
	 ++example_num) {
	for (const auto &x_pair : examples_x[example_num]) {
	    size_t index_x = x_pair.first;
	    double value_x = x_pair.second;
	    variance_x[index_x] += pow(value_x, 2);
	}
	for (const auto &y_pair : examples_y[example_num]) {
	    size_t index_y = y_pair.first;
	    double value_y = y_pair.second;
	    variance_y[index_y] += pow(value_y, 2);
	    for (const auto &x_pair : examples_x[example_num]) {
		size_t index_x = x_pair.first;
		double value_x = x_pair.second;
		covariance_xy[index_y][index_x] += value_x * value_y;
	    }
	}
    }

    // Compute CCA transformations from the unnormalized covariance values.
    PerformCCA(&covariance_xy, variance_x, variance_y);
}

void SparseCCASolver::LoadVariance(const string &variance_path,
				   unordered_map<size_t, double> *variance) {
    ifstream file(variance_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << variance_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    size_t index = 0;
    while (file.good()) {
	getline(file, line);  // <variance in the i-th dimension>
	if (line == "") { continue; }
	string_manipulator.split(line, " ", &tokens);
	ASSERT(tokens.size() == 1, "Bad format: " << line);
	(*variance)[index++] = stod(tokens[0]);
    }
}

double SparseCCASolver::ScaleCovariance(double value_xy,
					double value_x, double value_y) {
    double scaled_value_xy = value_xy;
    if (scaling_method_ == "cca") {
	scaled_value_xy /= sqrt(value_x + smoothing_term_);
	scaled_value_xy /= sqrt(value_y + smoothing_term_);
    } else if (scaling_method_ == "pmi") {
	scaled_value_xy = log(scaled_value_xy);
	scaled_value_xy += log(num_samples_);
	scaled_value_xy -= log(value_x);
	scaled_value_xy -= log(value_y);
	scaled_value_xy = max(scaled_value_xy, 0.0);
    } else {
	ASSERT(false, "Unknown scaling method: " << scaling_method_);
    }
    return scaled_value_xy;
}

void SparseCCASolver::ExtractCCAProjections(
    const SparseSVDSolver &svd_solver,
    const unordered_map<size_t, double> &variance_x,
    const unordered_map<size_t, double> &variance_y) {
    size_t dim_x = variance_x.size();
    size_t dim_y = variance_y.size();
    ASSERT(svd_solver.HasSVDResult(), "No SVD result for extraction.");
    ASSERT(svd_solver.left_singular_vectors()->rows == cca_dim_ &&
	   svd_solver.left_singular_vectors()->cols == dim_x &&
	   svd_solver.right_singular_vectors()->rows == cca_dim_ &&
	   svd_solver.right_singular_vectors()->cols == dim_y,
	   "Dimensions don't match between SVD and CCA.");

    // Store correlation values.
    cca_correlations_.resize(cca_dim_);
    for (size_t i = 0; i < cca_dim_; ++i) {
	cca_correlations_(i) = *(svd_solver.singular_values() + i);
    }

    // Set the view X projection as scaled left singular vectors.
    cca_transformation_x_.resize(cca_dim_, dim_x);
    for (size_t col = 0; col < dim_x; ++col) {
	double scale = sqrt(variance_x.at(col) + smoothing_term_);
	for (size_t row = 0; row < cca_dim_; ++row) {
	    cca_transformation_x_(row, col) =
		svd_solver.left_singular_vectors()->value[row][col];
	    if (scaling_method_ == "cca") {
		cca_transformation_x_(row, col) /= scale;
	    } else if (scaling_method_ == "pmi") {
		cca_transformation_x_(row, col) /= sqrt(cca_correlations_(row));
	    } else {
		ASSERT(false, "Unknown scaling method: " << scaling_method_);
	    }
	}
    }

    // Set the view Y projection as scaled right singular vectors.
    cca_transformation_y_.resize(cca_dim_, dim_y);
    for (size_t col = 0; col < dim_y; ++col) {
	double scale = sqrt(variance_y.at(col) + smoothing_term_);
	for (size_t row = 0; row < cca_dim_; ++row) {
	    cca_transformation_y_(row, col) =
		svd_solver.right_singular_vectors()->value[row][col];
	    if (scaling_method_ == "cca") {
		cca_transformation_y_(row, col) /= scale;
	    } else if (scaling_method_ == "pmi") {
		cca_transformation_y_(row, col) /= sqrt(cca_correlations_(row));
	    } else {
		ASSERT(false, "Unknown scaling method: " << scaling_method_);
	    }
	}
    }
}
