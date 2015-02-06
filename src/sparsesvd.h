// Author: Karl Stratos (karlstratos@gmail.com)
//
// This is a wrapper around SVDLIBC that provides a clean interface for
// performing singular value decomposition (SVD) on sparse matrices in
// Standard C++.

#ifndef SPARSESVD_H
#define SPARSESVD_H

#include <iostream>
#include <unordered_map>

#include "util.h"

extern "C" {  // For using C code from C++ code.
#include "../third_party/SVDLIBC/svdlib.h"
}

class SparseSVDSolver {
public:
    // Initializes an empty SVD solver.
    SparseSVDSolver() { }

    // Initializes with a sparse matrix in a file.
    SparseSVDSolver(const string &file_path) { LoadSparseMatrix(file_path); }

    // Cleans up memory at deletion.
    ~SparseSVDSolver();

    // Loads a sparse matrix from a text file into the class object.
    void LoadSparseMatrix(const string &file_path);

    // Reads a sparse matrix from a text file with the following format:
    //    <num_rows> <num_columns> <num_nonzeros>
    //    for each column from left to right:
    //       <num_nonzeros_in_col>
    //       for each (row_index, value) in this column:
    //          <row_index> <value>
    SMat ReadSparseMatrixFromFile(const string &file_path);

    // Writes a sparse matrix as a file.
    void WriteSparseMatrix(
	const unordered_map<size_t, unordered_map<size_t, double> >
	&column_map, const string file_path);

    // Loads a sparse matrix M for SVD: column_map[j][i] = M_{i,j}.
    void LoadSparseMatrix(
	const unordered_map<size_t, unordered_map<size_t, double> >
	&column_map);

    // Computes a thin SVD of the loaded sparse matrix.
    void SolveSparseSVD(size_t rank);

    // Returns the string form of the SVD calculation.
    string ToString();

    // Does it have some matrix loaded?
    bool HasMatrix() const { return sparse_matrix_ != nullptr; }

    // Does it have some SVD result?
    bool HasSVDResult() const { return svd_result_ != nullptr; }

    // Returns a pointer to the sparse matrix for SVD.
    SMat sparse_matrix() { return sparse_matrix_; }

    // Returns a pointer to a matrix whose i-th row is the left singular vector
    // corresponding to the i-th largest singular value.
    DMat left_singular_vectors() const { return svd_result_->Ut; }

    // Returns a pointer to a matrix whose i-th row is the right singular vector
    // corresponding to the i-th largest singular value.
    DMat right_singular_vectors() const { return svd_result_->Vt; }

    // Returns a pointer to computed singular values.
    double *singular_values() const { return svd_result_->S; }

    // Returns the rank of the computed SVD.
    size_t rank() const { return svd_result_->d; }

    // Frees the loaded sparse matrix and sets it to nullptr.
    void FreeSparseMatrix();

    // Frees the loaded SVD result and sets it to nullptr.
    void FreeSVDResult();

private:
    // Sparse matrix for SVD.
    SMat sparse_matrix_ = nullptr;

    // Result of the latest SVD computation.
    SVDRec svd_result_ = nullptr;
};

#endif  // SPARSESVD_H
