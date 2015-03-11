// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Optimizer of the weighted squared (WSQ) loss between each matrix value and
// the dot product of a corresponding pair of parameter vectors.

#ifndef WSQLOSS_H
#define WSQLOSS_H

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

typedef unordered_map<size_t, vector<tuple<size_t, double, double> > > WSQMap;

// Computes the Moore–Penrose pseudo-inverse of a given matrix.
Eigen::MatrixXd compute_pinv(const Eigen::MatrixXd &M);

// Solves a particular block of rows in V using fixed U.
void solve_rows(const Eigen::MatrixXd &U, size_t num_threads, size_t thread_num,
		const WSQMap &mapping, Eigen::MatrixXd *V);

class WSQLossOptimizer {
public:
    // Initializes empty.
    WSQLossOptimizer() { }

    ~WSQLossOptimizer() { }

    // Modifies given (U,V) to minimize weighted squared loss. Assumes the value
    // matrix A and the weight matrix W to be given in both the column- and the
    // row-major formats:
    //    col2row: c -> {(r, A_{r,c}, W_{r,c}): A_{r,c}, W_{r,c} > 0}
    //    row2col: r -> {(c, A_{r,c}, W_{r,c}): A_{r,c}, W_{r,c} > 0}
    void Optimize(const WSQMap &col2row, const WSQMap &row2col,
		  Eigen::MatrixXd *U, Eigen::MatrixXd *V);

    // Sets the number of threads.
    void set_num_threads(size_t num_threads) { num_threads_ = num_threads; }

    // Sets the flag for printing reports to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    // Computes the squared loss with given parameters.
    double ComputeLoss(const WSQMap &col2row, const Eigen::MatrixXd &U,
		       const Eigen::MatrixXd &V);

    // Minimum required loss reduction.
    const double kMinimumLossImprovement_ = 0.01;

    // Maximum number of training epochs.
    size_t max_num_epochs_ = 100;

    // Number of threads.
    size_t num_threads_ = 20;

    // Print reports to stderr?
    bool verbose_ = true;
};

#endif  // WSQLOSS_H
