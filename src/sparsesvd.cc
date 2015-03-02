// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "sparsesvd.h"

#include <fstream>
#include <iomanip>
#include <math.h>
#include <sstream>

SparseSVDSolver::~SparseSVDSolver() {
    FreeSparseMatrix();
    FreeSVDResult();
}

void SparseSVDSolver::LoadSparseMatrix(const string &file_path) {
    // Free the sparse matrix variable in case it's filled.
    FreeSparseMatrix();
    sparse_matrix_ = ReadSparseMatrixFromFile(file_path);
}

SMat SparseSVDSolver::ReadSparseMatrixFromFile(const string &file_path) {
    // Load from the sparse representation.
    vector<char> writable(file_path.begin(), file_path.end());
    writable.push_back('\0');
    SMat sparse_matrix = svdLoadSparseMatrix(&writable[0], 1);  // 1 = text
    return sparse_matrix;
}

void SparseSVDSolver::WriteSparseMatrix(
    const unordered_map<size_t, unordered_map<size_t, double> >
    &column_map, const string file_path) {
    // Compute the number of dimensions and nonzero values.
    size_t num_rows = 0;
    size_t num_columns = 0;
    size_t num_nonzeros = 0;
    for (const auto &col_pair: column_map) {
	size_t col = col_pair.first;
	if (col >= num_columns) { num_columns = col + 1; }
	for (const auto &row_pair: col_pair.second) {
	    size_t row = row_pair.first;
	    if (row >= num_rows) { num_rows = row + 1; }
	    ++num_nonzeros;
	}
    }

    // Write in the sparse text format for SVDLIBC.
    ofstream file(file_path, ios::out);
    ASSERT(file.is_open(), "Cannot open file: " << file_path);
    file << num_rows << " " << num_columns << " " << num_nonzeros << endl;
    for (size_t col = 0; col < num_columns; ++col) {
	if (column_map.find(col) == column_map.end()) {
	    file << 0 << endl;
	    continue;
	}
	file << column_map.at(col).size() << endl;
	for (const auto &row_pair: column_map.at(col)) {
	    file << row_pair.first << " " << row_pair.second << endl;
	}
    }
}

void SparseSVDSolver::LoadSparseMatrix(
    const unordered_map<size_t, unordered_map<size_t, double> > &column_map) {
    // Compute the number of dimensions and nonzero values.
    size_t num_rows = 0;
    size_t num_columns = 0;
    size_t num_nonzeros = 0;
    for (const auto &col_pair: column_map) {
	size_t col = col_pair.first;
	if (col >= num_columns) { num_columns = col + 1; }
	for (const auto &row_pair: col_pair.second) {
	    size_t row = row_pair.first;
	    if (row >= num_rows) { num_rows = row + 1; }
	    ++num_nonzeros;
	}
    }
    ASSERT(num_rows > 0 && num_columns > 0 && num_nonzeros > 1,
	   "SVDLIBC will not handle this matrix properly: "
	   << num_rows << " x " << num_columns << " with "
	   << num_nonzeros << " nonzeros?");

    // Keep track of nonzero values.
    size_t current_nonzero_index = 0;

    // Free the sparse matrix variable in case it's filled.
    FreeSparseMatrix();

    // Load the sparse matrix variable.
    sparse_matrix_ = svdNewSMat(num_rows, num_columns, num_nonzeros);
    for (size_t col = 0; col < num_columns; ++col) {
	sparse_matrix_->pointr[col] = current_nonzero_index;
	if (column_map.find(col) == column_map.end()) { continue; }
	for (const auto &row_pair: column_map.at(col)) {
	    size_t row = row_pair.first;
	    double value = row_pair.second;
	    sparse_matrix_->rowind[current_nonzero_index] = row;
	    sparse_matrix_->value[current_nonzero_index] = value;
	    ++current_nonzero_index;
	}
    }
    sparse_matrix_->pointr[num_columns] = num_nonzeros;
}

void SparseSVDSolver::SolveSparseSVD(size_t rank) {
    ASSERT(rank > 0, "SVD rank is given as <= 0: " << rank);
    ASSERT(HasMatrix(), "No matrix for SVD computation.");
    ASSERT(rank <= min(sparse_matrix_->rows, sparse_matrix_->cols), "SVD rank "
	   "is given as > min(num_rows, num_cols): " << rank << " > min("
	   << sparse_matrix_->rows << ", " << sparse_matrix_->cols << ")");

    // Free the current SVD result in case it's filled.
    FreeSVDResult();

    // Run the Lanczos algorithm with default parameters.
    svd_result_ = svdLAS2A(sparse_matrix_, rank);
}

void SparseSVDSolver::FreeSparseMatrix() {
    if (HasMatrix()) {
	svdFreeSMat(sparse_matrix_);
	sparse_matrix_ = nullptr;
    }
}

void SparseSVDSolver::FreeSVDResult() {
    if (HasSVDResult()) {
	svdFreeSVDRec(svd_result_);
	svd_result_ = nullptr;
    }
}
